from flask import Flask, request, jsonify
import numpy as np
from kmeans import KMeans
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Store the current state of the KMeans algorithm
kmeans_instance = None
data = None

@app.route('/api/init', methods=['POST'])
def initialize_kmeans():
    global kmeans_instance, data
    request_data = request.json
    points = np.array(request_data['points'])
    k = request_data['k']
    init_method = request_data['init_method']

    # Manual centroids passed in the request
    manual_centroids = request_data.get('manual_centroids')

    # Initialize KMeans with data and centroids if manual
    if init_method == 'manual' and manual_centroids:
        kmeans_instance = KMeans(data=points, k=k, init_method=init_method)
        kmeans_instance.centroids = np.array(manual_centroids)  # Set manually selected centroids
    else:
        kmeans_instance = KMeans(data=points, k=k, init_method=init_method)

    data = points

    # Run the first step of KMeans (only for non-manual or once manual centroids are set)
    centroids, assignments = kmeans_instance.fit(step_by_step=True)

    return jsonify({
        "centroids": centroids.tolist(),
        "assignments": list(map(int, assignments))
    })


@app.route('/api/step', methods=['POST'])
def step_kmeans():
    global kmeans_instance
    if not kmeans_instance:
        return jsonify({"error": "KMeans not initialized"}), 400

    # Perform the step for manual centroids or any other initialization
    centroids, assignments = kmeans_instance.fit(step_by_step=True)

    # Convert to JSON-friendly format
    return jsonify({
        "centroids": centroids.tolist(),
        "assignments": list(map(int, assignments))
    })


@app.route('/api/converge', methods=['POST'])
def converge_kmeans():
    global kmeans_instance
    # Perform KMeans until convergence for manual centroids or any other initialization
    if not kmeans_instance:
        return jsonify({"error": "KMeans not initialized"}), 400

    centroids, assignments = kmeans_instance.fit(step_by_step=False)

    return jsonify({
        "centroids": centroids.tolist(),
        "assignments": list(map(int, assignments))
    })



@app.route('/api/reset', methods=['POST'])
def reset_kmeans():
    global kmeans_instance
    # Reset KMeans state
    kmeans_instance = None
    return jsonify({"status": "KMeans reset successful"})

if __name__ == '__main__':
    app.run(debug=True, port=3001)
