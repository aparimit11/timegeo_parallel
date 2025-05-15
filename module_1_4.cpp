#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <memory>
#include <cstdio>
#include <sstream>
#include <algorithm>
#include <map>

#define pi 3.14159265358979323846
#define EPSILON 0.0000001
#define stay_dist_limit 0.3 // in km
#define min_home_count 10
#define min_work_count 50
#define second_home_ratio 1
#define departure_night_ratio 0.1
#define night_begin_time 88
#define min_home_work_dist 1
#define personal_limit 1
#define region_size (0.1*0.0095*3)
#define winterTimeStart 1667552400

int time_min=1659414864;
int time_max=1667260540;

struct RawRecord {
    int user_id;
    double lon;
    double lat;
    int timestamp;
};

struct StayPoint {
    double lon;
    double lat;
    int start_time;
    int duration;
};

struct StayRegionRecord {
    int region_id;
    int timestamp;
    int duration;
    double lon;
    double lat;
};

struct StayRegionHeader {
    int person_total_count;
    int stay_count;
    int region_count;
    int user_id;
};

struct FilteredStay {
    int counter;
    int time;
    int label;
    int user_id;
    int loc_id;
    double lon;
    double lat;
};

// --- Utility Functions ---

double deg2rad(double deg) { return (deg * pi / 180); }
double rad2deg(double rad) { return (rad * 180 / pi); }

double distance(double lat1, double lon1, double lat2, double lon2, char unit='K') {
    double theta, dist;
    theta = lon1 - lon2;
    dist = sin(deg2rad(lat1)) * sin(deg2rad(lat2)) + cos(deg2rad(lat1)) * cos(deg2rad(lat2)) * cos(deg2rad(theta));
    dist = acos(std::min(1.0, std::max(-1.0, dist))); // Clamp for safety
    dist = rad2deg(dist);
    dist = dist * 60 * 1.1515;
    switch(unit) {
        case 'M': break;
        case 'K': dist = dist * 1.609344; break;
        case 'N': dist = dist * 0.8684; break;
    }
    return (dist);
}

bool AreSame(double a, double b) { return fabs(a - b) < EPSILON; }

// --- Step 1: Read Parquet File ---

std::unordered_map<int, std::vector<RawRecord>> read_parquet(const std::string& filename) {
    std::unordered_map<int, std::vector<RawRecord>> user_records;
    
    try {
        std::cout << "Opening file: " << filename << std::endl;
        auto infile = arrow::io::ReadableFile::Open(filename).ValueOrDie();
        
        std::cout << "Creating ParquetFileReader..." << std::endl;
        std::unique_ptr<parquet::arrow::FileReader> reader;
        auto status = parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader);
        if (!status.ok()) {
            std::cerr << "Error opening Parquet file: " << status.ToString() << std::endl;
            return user_records;
        }
        
        // Get metadata
        std::shared_ptr<parquet::FileMetaData> metadata = reader->parquet_reader()->metadata();
        int64_t num_rows = metadata->num_rows();
        int num_row_groups = metadata->num_row_groups();
        
        std::cout << "File has " << num_rows << " rows in " << num_row_groups << " row groups" << std::endl;
        
        // Process in smaller chunks
        const int CHUNK_SIZE = 100000; // Process 100K rows at a time
        
        for (int row_group = 0; row_group < num_row_groups; row_group++) {
            std::cout << "Processing row group " << row_group + 1 << " of " << num_row_groups << std::endl;
            
            // Read this row group
            std::shared_ptr<arrow::Table> table;
            status = reader->ReadRowGroup(row_group, &table);
            if (!status.ok()) {
                std::cerr << "Error reading row group: " << status.ToString() << std::endl;
                continue;
            }
            
            int64_t rows_in_group = table->num_rows();
            std::cout << "Row group has " << rows_in_group << " rows" << std::endl;
            
            // Get column indices
            int ref_id_idx = -1, datetime_idx = -1, lat_idx = -1, lng_idx = -1;
            
            for (int i = 0; i < table->num_columns(); i++) {
                std::string name = table->field(i)->name();
                if (name == "ref_id") ref_id_idx = i;
                else if (name == "datetime") datetime_idx = i;
                else if (name == "lat") lat_idx = i;
                else if (name == "lng") lng_idx = i;
            }
            
            if (ref_id_idx == -1 || datetime_idx == -1 || lat_idx == -1 || lng_idx == -1) {
                std::cerr << "Missing required columns in row group" << std::endl;
                continue;
            }
            
            // Process this row group in chunks
            for (int64_t chunk_start = 0; chunk_start < rows_in_group; chunk_start += CHUNK_SIZE) {
                int64_t chunk_end = std::min(chunk_start + CHUNK_SIZE, rows_in_group);
                std::cout << "Processing chunk " << chunk_start/CHUNK_SIZE + 1 
                          << " of " << (rows_in_group + CHUNK_SIZE - 1)/CHUNK_SIZE 
                          << " (" << chunk_start << " to " << chunk_end << ")" << std::endl;
                
                // Get column arrays for this chunk
                auto ref_id_array = std::static_pointer_cast<arrow::StringArray>(table->column(ref_id_idx)->chunk(0));
                auto datetime_array = std::static_pointer_cast<arrow::DoubleArray>(table->column(datetime_idx)->chunk(0));
                auto lat_array = std::static_pointer_cast<arrow::DoubleArray>(table->column(lat_idx)->chunk(0));
                auto lng_array = std::static_pointer_cast<arrow::DoubleArray>(table->column(lng_idx)->chunk(0));
                
                // Process rows in this chunk
                std::unordered_map<int, std::vector<RawRecord>> chunk_records;
                
                for (int64_t i = chunk_start; i < chunk_end; i++) {
                    std::string ref_id = ref_id_array->GetString(i);
                    double datetime = datetime_array->Value(i);
                    double lat = lat_array->Value(i);
                    double lng = lng_array->Value(i);
                    
                    int user_id = std::hash<std::string>{}(ref_id) % 1000000;
                    int timestamp = static_cast<int>(datetime);
                    
                    chunk_records[user_id].push_back({user_id, lng, lat, timestamp});
                }
                
                // Merge chunk records into main map
                for (const auto& [user_id, records] : chunk_records) {
                    user_records[user_id].insert(
                        user_records[user_id].end(),
                        records.begin(),
                        records.end()
                    );
                }
                
                std::cout << "Processed chunk, total users so far: " << user_records.size() << std::endl;
            }
        }
        
        std::cout << "Finished reading Parquet file" << std::endl;
        std::cout << "Total users: " << user_records.size() << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in read_parquet: " << e.what() << std::endl;
    }
    
    return user_records;
}

// --- Step 2: Generate Stay Points ---

std::unordered_map<int, std::vector<StayPoint>> generate_stay_points(const std::unordered_map<int, std::vector<RawRecord>>& user_records) {
    std::unordered_map<int, std::vector<StayPoint>> user_stays;
    
    #pragma omp parallel
    {
        std::unordered_map<int, std::vector<StayPoint>> local_stays;
        
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < 1000000; i++) {
            if (user_records.find(i) == user_records.end()) continue;
            
            const auto& records = user_records.at(i);
            if (records.size() < 2) continue;
            
            std::vector<StayPoint> stays;
            
            // Sort records by timestamp
            std::vector<RawRecord> sorted_records = records;
            std::sort(sorted_records.begin(), sorted_records.end(), 
                     [](const RawRecord& a, const RawRecord& b) { return a.timestamp < b.timestamp; });
            
            // Find stay points
            int j = 0;
            while (j < sorted_records.size() - 1) {
                double avg_lat = sorted_records[j].lat;
                double avg_lon = sorted_records[j].lon;
                int count = 1;
                int start_time = sorted_records[j].timestamp;
                int k = j + 1;
                
                while (k < sorted_records.size() && 
                       distance(sorted_records[j].lat, sorted_records[j].lon, 
                                sorted_records[k].lat, sorted_records[k].lon) < stay_dist_limit) {
                    avg_lat = (avg_lat * count + sorted_records[k].lat) / (count + 1);
                    avg_lon = (avg_lon * count + sorted_records[k].lon) / (count + 1);
                    count++;
                    k++;
                }
                
                if (k > j + 1) {
                    int duration = sorted_records[k-1].timestamp - start_time;
                    if (duration > 0) {
                        stays.push_back({avg_lon, avg_lat, start_time, duration});
                    }
                    j = k;
                } else {
                    j++;
                }
            }
            
            if (!stays.empty()) {
                local_stays[i] = stays;
            }
        }
        
        #pragma omp critical
        {
            for (const auto& [user_id, stays] : local_stays) {
                user_stays[user_id] = stays;
            }
        }
    }
    
    return user_stays;
}

// --- Step 3: Generate Stay Regions ---

std::unordered_map<int, std::vector<StayRegionRecord>> generate_stay_regions(const std::unordered_map<int, std::vector<StayPoint>>& user_stays) {
    std::unordered_map<int, std::vector<StayRegionRecord>> user_regions;
    
    #pragma omp parallel
    {
        std::unordered_map<int, std::vector<StayRegionRecord>> local_regions;
        
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < 1000000; i++) {
            if (user_stays.find(i) == user_stays.end()) continue;
            
            const auto& stays = user_stays.at(i);
            if (stays.empty()) continue;
            
            // Cluster stay points into regions
            std::vector<std::vector<int>> clusters;
            std::vector<bool> visited(stays.size(), false);
            
            for (int j = 0; j < stays.size(); j++) {
                if (visited[j]) continue;
                
                std::vector<int> cluster;
                cluster.push_back(j);
                visited[j] = true;
                
                for (int k = 0; k < stays.size(); k++) {
                    if (visited[k]) continue;
                    
                    if (distance(stays[j].lat, stays[j].lon, stays[k].lat, stays[k].lon) < region_size) {
                        cluster.push_back(k);
                        visited[k] = true;
                    }
                }
                
                clusters.push_back(cluster);
            }
            
            // Create stay region records
            std::vector<StayRegionRecord> regions;
            
            for (int j = 0; j < clusters.size(); j++) {
                const auto& cluster = clusters[j];
                
                double avg_lat = 0, avg_lon = 0;
                for (int idx : cluster) {
                    avg_lat += stays[idx].lat;
                    avg_lon += stays[idx].lon;
                }
                avg_lat /= cluster.size();
                avg_lon /= cluster.size();
                
                for (int idx : cluster) {
                    regions.push_back({
                        j,                  // region_id
                        stays[idx].start_time,  // timestamp
                        stays[idx].duration,    // duration
                        avg_lon,            // lon
                        avg_lat             // lat
                    });
                }
            }
            
            if (!regions.empty()) {
                local_regions[i] = regions;
            }
        }
        
        #pragma omp critical
        {
            for (const auto& [user_id, regions] : local_regions) {
                user_regions[user_id] = regions;
            }
        }
    }
    
    return user_regions;
}

// --- Step 4: Filter Stay Regions ---

std::vector<FilteredStay> filter_stay_regions(const std::unordered_map<int, std::vector<StayRegionRecord>>& user_regions) {
    std::vector<FilteredStay> filtered;
    int counter = 0;
    
    #pragma omp parallel
    {
        std::vector<FilteredStay> local_filtered;
        int local_counter = 0;
        
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < 1000000; i++) {
            if (user_regions.find(i) == user_regions.end()) continue;
            
            const auto& regions = user_regions.at(i);
            if (regions.empty()) continue;
            
            // Group by region_id
            std::map<int, std::vector<StayRegionRecord>> region_groups;
            for (const auto& region : regions) {
                region_groups[region.region_id].push_back(region);
            }
            
            // Calculate home and work locations
            int place_count = region_groups.size();
            if (place_count < 2) continue;
            
            std::vector<double> place_lon(place_count), place_lat(place_count);
            std::vector<int> place_count_vec(place_count, 0);
            std::vector<int> work_count(place_count, 0);
            std::vector<int> home_count(place_count, 0);
            
            int idx = 0;
            for (const auto& [region_id, records] : region_groups) {
                place_lon[idx] = records[0].lon;
                place_lat[idx] = records[0].lat;
                place_count_vec[idx] = records.size();
                idx++;
            }
            
            // Extract all records for this user
            std::vector<double> lons, lats;
            std::vector<int> times, durations, locs, ids;
            
            for (const auto& [region_id, records] : region_groups) {
                for (const auto& record : records) {
                    lons.push_back(record.lon);
                    lats.push_back(record.lat);
                    times.push_back(record.timestamp);
                    durations.push_back(record.duration);
                    locs.push_back(record.region_id);
                    ids.push_back(i);
                }
            }
            
            int per_count = lons.size();
            
            // Sort by timestamp
            std::vector<size_t> indices(per_count);
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(),
                     [&times](size_t a, size_t b) { return times[a] < times[b]; });
            
            std::vector<double> sorted_lons(per_count), sorted_lats(per_count);
            std::vector<int> sorted_times(per_count), sorted_durations(per_count), sorted_locs(per_count), sorted_ids(per_count);
            
            for (int j = 0; j < per_count; j++) {
                sorted_lons[j] = lons[indices[j]];
                sorted_lats[j] = lats[indices[j]];
                sorted_times[j] = times[indices[j]];
                sorted_durations[j] = durations[indices[j]];
                sorted_locs[j] = locs[indices[j]];
                sorted_ids[j] = ids[indices[j]];
            }
            
            lons = sorted_lons;
            lats = sorted_lats;
            times = sorted_times;
            durations = sorted_durations;
            locs = sorted_locs;
            ids = sorted_ids;
            
            // Identify home and work
            int home = -1, work = -1;
            int home_valid_sign = 0, work_valid_sign = 0;
            double work_product = 0, home_work_dist = 0;
            int work_num = 0;
            
            // Count home and work visits
            for (int j = 0; j < per_count; j++) {
                int timeOff = (times[j] < winterTimeStart) ? 4*3600 : 5*3600;
                int eslot = ((times[j] - timeOff) % 86400) / 900;
                
                if (eslot >= 32 && eslot < 72) {
                    work_count[locs[j]]++;
                } else if (eslot >= 80 || eslot < 32) {
                    home_count[locs[j]]++;
                }
            }
            
            // Find home location
            int home_num = 0, second_home_num = 0;
            for (int j = 0; j < place_count; j++) {
                if (home_count[j] > home_num) {
                    second_home_num = home_num;
                    home_num = home_count[j];
                    home = j;
                    if (home_num >= min_home_count) home_valid_sign = 1;
                    else home_valid_sign = 0;
                } else if (home_count[j] > second_home_num) {
                    second_home_num = home_count[j];
                }
            }
            if (second_home_num > home_num * second_home_ratio) home_valid_sign = 0;

            if (home_valid_sign == 1) {
                int night_departure_count = 0;
                for (int j = 0; j < per_count; j++) {
                    int timeOff = (times[j] < winterTimeStart) ? 4*3600 : 5*3600;
                    if (locs[j] != home) {
                        int eslot = ((((int)(times[j] + durations[j])) - timeOff) % 86400) / 900;
                        if ((eslot > night_begin_time)) night_departure_count++;
                    }
                }
                if (night_departure_count > per_count * departure_night_ratio) home_valid_sign = 0;

                for (int j = 0; j < place_count; j++) {
                    if (j != home && work_count[j] * distance(place_lat[home], place_lon[home], place_lat[j], place_lon[j], 'K') > work_product) {
                        work_product = work_count[j] * distance(place_lat[home], place_lon[home], place_lat[j], place_lon[j], 'K');
                        work_num = work_count[j];
                        home_work_dist = distance(place_lat[home], place_lon[home], place_lat[j], place_lon[j], 'K');
                        work = j;
                        if (work_num >= min_work_count && home_work_dist > min_home_work_dist) work_valid_sign = 1;
                        else work_valid_sign = 0;
                    }
                }
            }
            if (!work_valid_sign) work = -1;

            // Merge consecutive same activities
            std::vector<double> one_day_lons, one_day_lats;
            std::vector<int> one_day_times, one_day_durations, one_day_loc_id, one_day_ids;
            for (int j = 0; j < per_count - 1; j++) {
                if (one_day_lons.empty() || locs[j] != locs[j - 1]) {
                    one_day_lons.push_back(lons[j]);
                    one_day_lats.push_back(lats[j]);
                    one_day_times.push_back(times[j]);
                    one_day_durations.push_back(times[j+1] - times[j]);
                    one_day_loc_id.push_back(locs[j]);
                    one_day_ids.push_back(ids[j]);
                } else {
                    one_day_durations[one_day_lons.size() - 1] = times[j] - one_day_times[one_day_lons.size() - 1];
                }
            }
            int one_day_count = one_day_lons.size();

            for (int j = 0; j < one_day_count; j++) {
                int label = 0;
                if (one_day_loc_id[j] == home) label = 1;
                else if (one_day_loc_id[j] == work) label = 2;
                local_filtered.push_back({local_counter, one_day_times[j], label, one_day_ids[j], one_day_loc_id[j], one_day_lons[j], one_day_lats[j]});
            }
            
            local_counter++;
        }
        
        #pragma omp critical
        {
            // Merge local results into global vector
            filtered.insert(filtered.end(), local_filtered.begin(), local_filtered.end());
            counter += local_counter;
        }
    }
    
    return filtered;
}

// --- Step 5: Write Output ---

void write_output(const std::vector<FilteredStay>& filtered, const std::string& filename) {
    FILE* fout = fopen(filename.c_str(), "w");
    for (const auto& stay : filtered) {
        fprintf(fout, "%d %d %d %d %d %f %f\n",
            stay.counter, stay.time, stay.label, stay.user_id, stay.loc_id, stay.lon, stay.lat);
    }
    fclose(fout);
}

// --- Process in Streaming Mode ---

void process_streaming(const std::string& input_file, const std::string& output_file) {
    try {
        std::cout << "Opening file: " << input_file << std::endl;
        auto infile = arrow::io::ReadableFile::Open(input_file).ValueOrDie();
        
        std::cout << "Creating ParquetFileReader..." << std::endl;
        std::unique_ptr<parquet::arrow::FileReader> reader;
        auto status = parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader);
        if (!status.ok()) {
            std::cerr << "Error opening Parquet file: " << status.ToString() << std::endl;
            return;
        }
        
        // Get metadata
        std::shared_ptr<parquet::FileMetaData> metadata = reader->parquet_reader()->metadata();
        int64_t num_rows = metadata->num_rows();
        int num_row_groups = metadata->num_row_groups();
        
        std::cout << "File has " << num_rows << " rows in " << num_row_groups << " row groups" << std::endl;
        
        // Open output file
        FILE* fout = fopen(output_file.c_str(), "w");
        if (!fout) {
            std::cerr << "Failed to open output file: " << output_file << std::endl;
            return;
        }
        
        // Process in smaller chunks
        const int CHUNK_SIZE = 100000; // Process 100K rows at a time
        int total_users_processed = 0;
        int global_counter = 0;
        
        for (int row_group = 0; row_group < num_row_groups; row_group++) {
            std::cout << "Processing row group " << row_group + 1 << " of " << num_row_groups << std::endl;
            
            // Read this row group
            std::shared_ptr<arrow::Table> table;
            status = reader->ReadRowGroup(row_group, &table);
            if (!status.ok()) {
                std::cerr << "Error reading row group: " << status.ToString() << std::endl;
                continue;
            }
            
            int64_t rows_in_group = table->num_rows();
            std::cout << "Row group has " << rows_in_group << " rows" << std::endl;
            
            // Get column indices
            int ref_id_idx = -1, datetime_idx = -1, lat_idx = -1, lng_idx = -1;
            
            for (int i = 0; i < table->num_columns(); i++) {
                std::string name = table->field(i)->name();
                if (name == "ref_id") ref_id_idx = i;
                else if (name == "datetime") datetime_idx = i;
                else if (name == "lat") lat_idx = i;
                else if (name == "lng") lng_idx = i;
            }
            
            if (ref_id_idx == -1 || datetime_idx == -1 || lat_idx == -1 || lng_idx == -1) {
                std::cerr << "Missing required columns in row group" << std::endl;
                continue;
            }
            
            // Process this row group in chunks
            for (int64_t chunk_start = 0; chunk_start < rows_in_group; chunk_start += CHUNK_SIZE) {
                int64_t chunk_end = std::min(chunk_start + CHUNK_SIZE, rows_in_group);
                std::cout << "Processing chunk " << chunk_start/CHUNK_SIZE + 1 
                          << " of " << (rows_in_group + CHUNK_SIZE - 1)/CHUNK_SIZE 
                          << " (" << chunk_start << " to " << chunk_end << ")" << std::endl;
                
                // Get column arrays for this chunk
                auto ref_id_array = std::static_pointer_cast<arrow::StringArray>(table->column(ref_id_idx)->chunk(0));
                auto datetime_array = std::static_pointer_cast<arrow::DoubleArray>(table->column(datetime_idx)->chunk(0));
                auto lat_array = std::static_pointer_cast<arrow::DoubleArray>(table->column(lat_idx)->chunk(0));
                auto lng_array = std::static_pointer_cast<arrow::DoubleArray>(table->column(lng_idx)->chunk(0));
                
                // Process rows in this chunk
                std::unordered_map<int, std::vector<RawRecord>> chunk_records;
                
                for (int64_t i = chunk_start; i < chunk_end; i++) {
                    std::string ref_id = ref_id_array->GetString(i);
                    double datetime = datetime_array->Value(i);
                    double lat = lat_array->Value(i);
                    double lng = lng_array->Value(i);
                    
                    int user_id = std::hash<std::string>{}(ref_id) % 1000000;
                    int timestamp = static_cast<int>(datetime);
                    
                    chunk_records[user_id].push_back({user_id, lng, lat, timestamp});
                }
                
                std::cout << "Processing " << chunk_records.size() << " users in this chunk" << std::endl;
                
                // Generate stay points for this chunk
                auto chunk_stays = generate_stay_points(chunk_records);
                
                // Generate stay regions for this chunk
                auto chunk_regions = generate_stay_regions(chunk_stays);
                
                // Filter stay regions for this chunk
                auto chunk_filtered = filter_stay_regions(chunk_regions);
                
                // Write results for this chunk
                for (const auto& stay : chunk_filtered) {
                    fprintf(fout, "%d %d %d %d %d %f %f\n",
                        global_counter++, stay.time, stay.label, stay.user_id, stay.loc_id, stay.lon, stay.lat);
                }
                
                total_users_processed += chunk_records.size();
                std::cout << "Processed " << chunk_records.size() << " users in this chunk, " 
                          << total_users_processed << " users total so far" << std::endl;
                
                // Clear memory for this chunk
                chunk_records.clear();
                chunk_stays.clear();
                chunk_regions.clear();
                chunk_filtered.clear();
            }
        }
        
        fclose(fout);
        std::cout << "Finished processing all chunks, total users: " << total_users_processed << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in process_streaming: " << e.what() << std::endl;
    }
}

// --- Main ---

int main() {
    std::string parquet_file = "/home/aparimit/resilience/processed_files/input_parquet/20200120_132814_01037_7dapg_0761f0b9-ff26-475f-8af2-084e1b22b969.parquet";
    std::string output_file = "final_output.txt";

    std::cout << "Reading parquet file..." << std::endl;
    auto user_records = read_parquet(parquet_file);

    std::cout << "Generating stay points..." << std::endl;
    auto user_stays = generate_stay_points(user_records);

    std::cout << "Generating stay regions..." << std::endl;
    auto user_regions = generate_stay_regions(user_stays);

    std::cout << "Filtering stay regions..." << std::endl;
    auto filtered = filter_stay_regions(user_regions);

    std::cout << "Writing output..." << std::endl;
    write_output(filtered, output_file);

    std::cout << "Done." << std::endl;
    return 0;
}
