import pandas as pd
import networkx as nx
import numpy as np
from multiprocessing import Pool, cpu_count
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

def compute_shortest_paths_for_graph(max_distance_km=None, sample_nodes=None, use_scipy=True):
    """
    Ultra-fast shortest paths with multiple speed optimizations

    Args:
        max_distance_km: Only compute paths up to this distance (None = all paths)
        sample_nodes: Only use this many random nodes (None = all nodes)
        use_scipy: Use scipy's C++ implementation (much faster)
    """

    print("🚀 Loading data...")
    start = time.time()

    file_name = 'data_Maastricht_2024.xlsx'
    nodes_df = pd.read_excel(file_name, 'Nodes')
    nodes_df['NODE ID'] = nodes_df['NODE ID'].astype(int)

    edges_df = pd.read_excel(file_name, 'Edges')
    edges_df['V1'] = edges_df['V1'].astype(int)
    edges_df['V2'] = edges_df['V2'].astype(int)

    # Always build an undirected graph for walking
    G = nx.Graph()
    G.add_nodes_from(nodes_df['NODE ID'])

    # Prepare edges
    edges_data = []
    speed_mps = 5.0   # example walking speed (5 km/h)

    for _, row in edges_df.iterrows():
        u, v = row['V1'], row['V2']
        if G.has_node(u) and G.has_node(v):
            dist_m = row['DIST']
            time_s = dist_m / speed_mps if dist_m > 0 else 0
            # Regardless of the "DIRECTED" flag, add edges for both directions
            edges_data.append((u, v, dist_m, time_s))
            edges_data.append((v, u, dist_m, time_s))

    G.add_weighted_edges_from([(u, v, w) for u, v, w, _ in edges_data])

    print(f"📊 Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges ({time.time()-start:.1f}s)")

    if use_scipy:
        # OPTIMIZATION 3: Use scipy's ultra-fast C++ Dijkstra
        print("⚡ Using scipy's C++ implementation...")
        return compute_with_scipy(G, nodes_df['NODE ID'].tolist(), edges_data, max_distance_km)
    else:
        # OPTIMIZATION 4: Parallel NetworkX (slower but more memory efficient)
        return compute_with_networkx_parallel(G, max_distance_km)

def compute_with_scipy(G, nodes, edges_data, max_distance_km):
    """Ultra-fast using scipy's C++ implementation"""
    start = time.time()

    # Create node mapping
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)

    # Build sparse matrices for distance and time
    rows, cols, dist_weights, time_weights = [], [], [], []

    for u, v, dist_m, time_s in edges_data:
        if u in node_to_idx and v in node_to_idx:
            i, j = node_to_idx[u], node_to_idx[v]
            rows.extend([i, j])
            cols.extend([j, i])
            dist_weights.extend([dist_m, dist_m])
            time_weights.extend([time_s, time_s])

    # Create sparse matrices
    dist_matrix = csr_matrix((dist_weights, (rows, cols)), shape=(n, n))
    time_matrix = csr_matrix((time_weights, (rows, cols)), shape=(n, n))

    print("🔥 Computing ALL shortest paths with scipy (C++ speed)...")

    # Compute all-pairs shortest paths (BLAZING FAST!)
    dist_paths = dijkstra(dist_matrix, directed=False)
    time_paths = dijkstra(time_matrix, directed=False)

    print(f"✅ Scipy computation done in {time.time()-start:.1f}s")

    # Create distance matrix DataFrame
    print("📝 Creating distance matrix CSV...")
    start_csv = time.time()

    # Apply distance cutoff if specified
    if max_distance_km:
        dist_paths[dist_paths > max_distance_km] = np.inf

    # Create upper triangular matrix (no mirrored paths)
    upper_tri_mask = np.triu(np.ones((n, n), dtype=bool), k=1)  # k=1 excludes diagonal
    
    # Create DataFrame with only upper triangular values
    distance_matrix_df = pd.DataFrame(
        dist_paths,
        index=nodes,
        columns=nodes
    )
    
    # Set lower triangle and diagonal to NaN to save space
    distance_matrix_df = distance_matrix_df.where(upper_tri_mask, np.nan)

    # Save the matrix
    distance_matrix_df.to_csv('distance_matrix.csv')
    print(f"📁 Distance matrix saved: {n}×{n} matrix (upper triangular) in {time.time()-start_csv:.1f}s")
    
    # Also create the original format for compatibility (only upper triangle pairs)
    if max_distance_km:
        max_dist_m = max_distance_km
        mask = (dist_paths <= max_dist_m) & upper_tri_mask
    else:
        mask = (~np.isinf(dist_paths)) & upper_tri_mask

    valid_pairs = np.where(mask)
    
    df = pd.DataFrame({
        'source': [nodes[i] for i in valid_pairs[0]],
        'target': [nodes[j] for j in valid_pairs[1]],
        'distance_m': dist_paths[valid_pairs].flatten(),
        'time_s': time_paths[valid_pairs].flatten()
    })

    df.to_csv('shortest_paths_optimized.csv', index=False)
    print(f"📁 Original format also saved: {len(df):,} rows (no duplicates)")
    
    return distance_matrix_df

def compute_with_networkx_parallel(G, max_distance_km):
    """Parallel NetworkX version (fallback)"""
    n_cores = cpu_count()
    nodes = list(G.nodes())
    chunk_size = len(nodes) // n_cores + 1
    chunks = [nodes[i:i+chunk_size] for i in range(0, len(nodes), chunk_size)]

    def process_chunk_with_cutoff(args):
        G, sources, cutoff = args
        results = {}
        cutoff_m = cutoff * 1000 if cutoff else None

        for source in sources:
            try:
                lengths = nx.single_source_dijkstra_path_length(
                    G, source, cutoff=cutoff_m, weight='weight'
                )
                results[source] = lengths
            except:
                results[source] = {}
        return results

    print(f"🔄 Using {n_cores} cores with NetworkX...")
    start = time.time()

    with Pool(n_cores) as pool:
        args = [(G, chunk, max_distance_km) for chunk in chunks]
        results = pool.map(process_chunk_with_cutoff, args)

    # Merge and create DataFrame
    all_paths = {}
    for result in results:
        all_paths.update(result)

    pairs = [(s, t, dist) for s, paths in all_paths.items()
             for t, dist in paths.items() if s != t]

    df = pd.DataFrame(pairs, columns=['source', 'target', 'distance_m'])
    df['time_s'] = df['distance_m'] / (5.0)

    df.to_csv('shortest_paths_optimized-0.5.csv', index=False)
    print(f"✅ Done in {time.time()-start:.1f}s, {len(df):,} rows")
    return df

# SPEED OPTIONS - Choose your optimization level:

# OPTION 1: Maximum speed - sample 2000 nodes, 10km max distance
# result = compute_shortest_paths_for_graph(max_distance_km=10, sample_nodes=2000)

# OPTION 2: Balanced - all nodes, 20km max distance
result = compute_shortest_paths_for_graph(max_distance_km=3000)

# OPTION 3: Full computation with scipy speed
# result = compute_shortest_paths_for_graph()


print(f"🎉 Final result: {len(result):,} node pairs computed!")