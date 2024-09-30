import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import { Container, TextField, Button, MenuItem, Select, InputLabel, Typography, Box, Grid, Alert } from '@mui/material';

const App = () => {
  const [points, setPoints] = useState([]);
  const [centroids, setCentroids] = useState([]);
  const [assignments, setAssignments] = useState([]);
  const [initMethod, setInitMethod] = useState("random");
  const [isInitialized, setIsInitialized] = useState(false);
  const [hasConverged, setHasConverged] = useState(false);
  const [k, setK] = useState("");
  const [manualCentroids, setManualCentroids] = useState([]);
  const [canInitialize, setCanInitialize] = useState(true); // Flag to control Initialize button

  useEffect(() => {
    generateNewDataset();
  }, []);

  // Generate a new random dataset
  const generateNewDataset = () => {
    const newPoints = Array.from({ length: 300 }, () => [
      Math.random() * 20 - 10,
      Math.random() * 20 - 10,
    ]);
    setPoints(newPoints);
    resetKMeans();
  };

  // Handle plot click for manual centroid selection
  const handlePlotClick = (event) => {
    if (initMethod === 'manual' && manualCentroids.length < parseInt(k)) {
      const { x, y } = event.points[0];
      setManualCentroids([...manualCentroids, [x, y]]);
      if (manualCentroids.length + 1 === parseInt(k)) {
        setIsInitialized(true);
        setCanInitialize(false); // Disable Initialize button after all centroids are selected
      }
    }
  };

  // Initialize KMeans
  const initializeKMeans = async () => {
    if (!k || isNaN(k) || k <= 0) {
      alert("Please enter a valid number of clusters (k).");
      return;
    }

    if (initMethod === 'manual') {
      setCentroids(manualCentroids);
      setAssignments([]); // Clear previous assignments
      setIsInitialized(true);
      setCanInitialize(false); // Disable Initialize button after manual initialization
    } else {
      try {
        const response = await axios.post('http://localhost:3001/api/init', {
          points: points,
          k: parseInt(k),
          init_method: initMethod
        });
        setCentroids(response.data.centroids);
        setAssignments(response.data.assignments);
        setIsInitialized(true);   // Allow stepping through the algorithm
        setHasConverged(false);    // Reset convergence status
      } catch (error) {
        console.error("Initialization error:", error);
        alert('Failed to initialize KMeans. Please try again.');
      }
    }
  };

  // Step through KMeans
  const stepKMeans = async () => {
    if (!isInitialized) {
      alert('KMeans not initialized. Please initialize first.');
      return;
    }

    try {
      const response = await axios.post('http://localhost:3001/api/step');
      setCentroids(response.data.centroids);
      setAssignments(response.data.assignments);
      if (response.data.has_converged) {
        setHasConverged(true);
      }
    } catch (error) {
      console.error("Step error:", error);
      alert('Failed to step KMeans. Please try again.');
    }
  };

  // Converge KMeans
  const convergeKMeans = async () => {
    if (!isInitialized) {
      alert('KMeans not initialized. Please initialize first.');
      return;
    }

    try {
      const response = await axios.post('http://localhost:3001/api/converge');
      setCentroids(response.data.centroids);
      setAssignments(response.data.assignments);
      setHasConverged(true);
    } catch (error) {
      console.error("Converge error:", error);
      alert('Failed to converge KMeans. Please try again.');
    }
  };

  // Reset KMeans
  const resetKMeans = async () => {
    try {
      await axios.post('http://localhost:3001/api/reset');
      setCentroids([]);
      setAssignments([]);
      setManualCentroids([]);
      setIsInitialized(false);
      setCanInitialize(true);  // Enable Initialize button after reset
      setHasConverged(false);
    } catch (error) {
      console.error("Reset error:", error);
      alert('Failed to reset KMeans. Please try again.');
    }
  };

  return (
    <Container maxWidth="md" style={{ marginTop: '20px' }}>
      <Typography variant="h3" gutterBottom align="center">
        KMeans Clustering Algorithm
      </Typography>

      <Box mb={4}>
        <Grid container spacing={3} alignItems="center">
          <Grid item xs={6}>
            <TextField
              label="Number of Clusters (k)"
              variant="outlined"
              fullWidth
              value={k}
              onChange={(e) => {
                setK(e.target.value);
                setManualCentroids([]);
                setIsInitialized(false);
                setCanInitialize(true);
              }}
              placeholder="Enter number of clusters"
            />
          </Grid>

          <Grid item xs={6}>
            <InputLabel id="init-method-label">Initialization Method</InputLabel>
            <Select
              labelId="init-method-label"
              value={initMethod}
              onChange={(e) => {
                setInitMethod(e.target.value);
                setManualCentroids([]);
                setIsInitialized(e.target.value !== 'manual'); // Allow init directly unless manual
                setCanInitialize(true);
              }}
              fullWidth
              variant="outlined"
            >
              <MenuItem value="random">Random</MenuItem>
              <MenuItem value="farthest_first">Farthest First</MenuItem>
              <MenuItem value="kmeans++">KMeans++</MenuItem>
              <MenuItem value="manual">Manual</MenuItem>
            </Select>
          </Grid>
        </Grid>
      </Box>

      <Box mb={4} textAlign="center">
        <Grid container spacing={2} justifyContent="center">
          <Grid item>
            <Button
              variant="contained"
              color="primary"
              onClick={initializeKMeans}
              disabled={!canInitialize || (initMethod === 'manual' && manualCentroids.length < parseInt(k))}
            >
              Initialize KMeans
            </Button>
          </Grid>
          <Grid item>
            <Button
              variant="contained"
              onClick={stepKMeans}
              disabled={!isInitialized || hasConverged}
            >
              Step Through KMeans
            </Button>
          </Grid>
          <Grid item>
            <Button
              variant="contained"
              color="success"
              onClick={convergeKMeans}
              disabled={!isInitialized || hasConverged}
            >
              Run to Convergence
            </Button>
          </Grid>
          <Grid item>
            <Button
              variant="contained"
              color="secondary"
              onClick={generateNewDataset}
            >
              Generate New Dataset
            </Button>
          </Grid>
          <Grid item>
            <Button
              variant="outlined"
              onClick={resetKMeans}
            >
              Reset Algorithm
            </Button>
          </Grid>
        </Grid>
      </Box>

      {/* Display success alert when KMeans has converged */}
      {hasConverged && <Alert severity="success">KMeans has converged!</Alert>}

      {/* Centering the Plot */}
      <Box display="flex" justifyContent="center" alignItems="center">
        <Plot
          data={[
            {
              x: points.map(point => point[0]),
              y: points.map(point => point[1]),
              mode: 'markers',
              marker: {
                color: assignments.map(a => ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][a]),
                size: 8,
              },
              type: 'scatter'
            },
            {
              x: centroids.map(centroid => centroid[0]),
              y: centroids.map(centroid => centroid[1]),
              mode: 'markers',
              marker: { color: 'red', size: 16, symbol: 'x' },
              type: 'scatter'
            },
            {
              x: manualCentroids.map(c => c[0]),
              y: manualCentroids.map(c => c[1]),
              mode: 'markers',
              marker: { color: 'green', size: 16, symbol: 'x' },
              type: 'scatter'
            }
          ]}
          layout={{ title: 'KMeans Clustering Data', autosize: true }}
          onClick={handlePlotClick}  // Handle click event on plot
        />
      </Box>
    </Container>
  );
};

export default App;
