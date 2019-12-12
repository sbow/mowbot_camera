// k-means-img.cpp
// 2019/12/11
// Shaun Bowman
//
// From https://github.com/goldsborough/k-means/blob/master/cpp/k-means.cpp
// Also http://www.goldsborough.me/c++/python/cuda/2017/09/10/20-32-46-exploring_k-means_in_python,_c++_and_cuda/
//
// Compile with: clang++ -std=c++11 -O3 -o k-means-img k-means-img.cpp
// ^note: ...ohhh3... not zero3
// sudo yum install clang
//
// Data structure:
// rrr,ggg,bbb
// rrr,ggg,bbb
// ^for each pixel, will perform k-means clustering by color
//
// Would be interesting in the future to also look at rr,gg,bb,xx,yy search-space


#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <vector>

struct Pixel {
  float r{0}, g{0}, b{0};
};

using DataFrame = std::vector<Pixel>;

float square(float value) {
  return value * value;
}

float squared_l2_distance(Pixel first, Pixel second) {
  return square(first.r - second.r) + square(first.g - second.g) + square(first.b - second.b);
}

DataFrame k_means(const DataFrame& data,
                  size_t k,
                  size_t number_of_iterations) {
  static std::random_device seed;
  static std::mt19937 random_number_generator(seed());
  std::uniform_int_distribution<size_t> indices(0, data.size() - 1);

  // Pick centroids as random points from the dataset.
  DataFrame means(k);
  for (auto& cluster : means) {
    cluster = data[indices(random_number_generator)];
  }

  std::vector<size_t> assignments(data.size());
  for (size_t iteration = 0; iteration < number_of_iterations; ++iteration) {
    // Find assignments.
    for (size_t point = 0; point < data.size(); ++point) {
      auto best_distance = std::numeric_limits<float>::max();
      size_t best_cluster = 0;
      for (size_t cluster = 0; cluster < k; ++cluster) {
        const float distance =
            squared_l2_distance(data[point], means[cluster]);
        if (distance < best_distance) {
          best_distance = distance;
          best_cluster = cluster;
        }
      }
      assignments[point] = best_cluster;
    }

    // Sum up and count points for each cluster.
    DataFrame new_means(k);
    std::vector<size_t> counts(k, 0);
    for (size_t pixel = 0; pixel < data.size(); ++pixel) {
      const auto cluster = assignments[pixel];
      new_means[cluster].r += data[pixel].r;
      new_means[cluster].g += data[pixel].g;
      new_means[cluster].b += data[pixel].b;
      counts[cluster] += 1;
    }

    // Divide sums by counts to get new centroids.
    for (size_t cluster = 0; cluster < k; ++cluster) {
      // Turn 0/0 into 0/1 to avoid zero division.
      const auto count = std::max<size_t>(1, counts[cluster]);
      means[cluster].r = new_means[cluster].r / count;
      means[cluster].g = new_means[cluster].g / count;
      means[cluster].b = new_means[cluster].b / count;
    }
  }

  return means;
}

int main(int argc, const char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage: k_means <data-file> <k> [iterations] [runs]"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const auto k = std::atoi(argv[2]);
  const auto iterations = (argc >= 4) ? std::atoi(argv[3]) : 300;
  const auto number_of_runs = (argc >= 5) ? std::atoi(argv[4]) : 10;

  DataFrame data;
  std::ifstream stream(argv[1]);
  if (!stream) {
    std::cerr << "Could not open file: " << argv[1] << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::string line;
  while (std::getline(stream, line)) {
    Pixel pixel;
    std::istringstream line_stream(line);
    size_t label;
    line_stream >> pixel.r >> pixel.g >> pixel.b;
    data.push_back(pixel);
  }

  DataFrame means;
  double total_elapsed = 0;
  for (int run = 0; run < number_of_runs; ++run) {
    const auto start = std::chrono::high_resolution_clock::now();
    means = k_means(data, k, iterations);
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    total_elapsed += duration.count();
  }
  std::cerr << "Took: " << total_elapsed / number_of_runs << "s ("
            << number_of_runs << " runs)" << std::endl;

  for (auto& mean : means) {
    std::cout << mean.r << " " << mean.g << " " << mean.b << std::endl;
  }
}