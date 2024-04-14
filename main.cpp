#include <iostream>
#include <vector>
#include <limits>
#include <mpi.h>
#include <chrono>
#include <cstdlib>
#include <ctime>

#define INF std::numeric_limits<int>::max()

// Define a structure to represent an edge
struct Edge {
    int src, dest, weight;
};

// Function to find the vertex with minimum key value
int minKey(const std::vector<int>& key, const std::vector<bool>& mstSet) {
    int min = INF, min_index = -1;
    for (int v = 0; v < key.size(); ++v) {
        if (!mstSet[v] && key[v] < min) {
            min = key[v];
            min_index = v;
        }
    }
    return min_index;
}

// Function to generate a random adjacency matrix for a graph with n vertices
std::vector<std::vector<int>> generateRandomGraph(int n) {
    std::vector<std::vector<int>> graph(n, std::vector<int>(n, 0)); // Initialize the matrix with zeros

    // Seed the random number generator with current time
    std::srand(12345); // Seed with a constant for reproducibility (remove for true randomness)

    // Generate random weights for edges
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) { // Skip diagonal and upper triangle (assuming undirected graph)
            int weight = std::rand() % 100; // Generate a random weight between 0 and 99
            graph[i][j] = weight;
            graph[j][i] = weight; // Assuming undirected graph, set symmetric weight
        }
    }

    return graph;
}

// Function to construct and print the MST using Prim's algorithm
void primMST(const std::vector<std::vector<int>>& graph, int numVertices) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> parent(numVertices);  // Array to store constructed MST
    std::vector<int> key(numVertices, INF); // Key values used to pick minimum weight edge in cut
    std::vector<bool> mstSet(numVertices, false); // To represent set of vertices not yet included in MST

    // Initialize the source vertex as the first vertex
    if (rank == 0) {
        key[0] = 0;
        parent[0] = -1; // First node is always the root of MST
    }

    // Start the timer
    auto start_time = std::chrono::high_resolution_clock::now();

    // Find the minimum key vertex not yet included in MST
    for (int count = 0; count < numVertices - 1; ++count) {
        int u = minKey(key, mstSet);

        // Broadcast the minimum key vertex to all processes
        MPI_Bcast(&u, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Mark the picked vertex as processed
        mstSet[u] = true;

        // Update key value and parent index of the adjacent vertices
        for (int v = 0; v < numVertices; ++v) {
            if (graph[u][v] && !mstSet[v] && graph[u][v] < key[v]) {
                parent[v] = u;
                key[v] = graph[u][v];
            }
        }
    }

    // Stop the timer
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    double runtime = duration.count() / 1000000.0;

    // Print the total weight of the MST
    if (rank == 0) {
        int totalWeight = 0;
        for (int i = 1; i < numVertices; ++i) {
            totalWeight += graph[i][parent[i]];
        }
        std::cout << "Total weight of the MST: " << totalWeight << std::endl;
    }

/*     // Print the constructed MST
    if (rank == 0) {
        std::cout << "Edge   Weight\n";
        for (int i = 1; i < numVertices; ++i) {
            std::cout << parent[i] << " - " << i << "    " << graph[i][parent[i]] << std::endl;
        }
    } */

    // Print the runtime
    if (rank == 0) {
        std::cout << "Runtime: " << runtime << " seconds\n";
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Define the number of vertices
    const int numVertices = 5000;

    // Generate a random graph
    std::vector<std::vector<int>> graph = generateRandomGraph(numVertices);

    // Run Prim's algorithm to find the MST
    primMST(graph, numVertices);

    MPI_Finalize();
    return 0;
}
