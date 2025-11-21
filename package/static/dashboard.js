let overallMetricsChart;
let predictionCharts = {}; // Store Chart.js instances for predictions (used in modal)
let modalChartInstance = null; // Store the Chart.js instance for the modal

document.addEventListener("DOMContentLoaded", async () => {
  // Fetch and display global metrics
  await fetchOverallMetrics();

  // Initial population of routes and zones for the pre-selected city (always using latest round)
  const initialCitySelect = document.getElementById("city-select");
  if (initialCitySelect && initialCitySelect.value) {
    await populateRouteAndZoneFilters(initialCitySelect.value);
  }

  // Set default datetime to current time for convenience
  const now = new Date();
  const year = now.getFullYear();
  const month = (now.getMonth() + 1).toString().padStart(2, "0");
  const day = now.getDate().toString().padStart(2, "0");
  const hours = now.getHours().toString().padStart(2, "0");
  const minutes = now.getMinutes().toString().padStart(2, "0");
  document.getElementById(
    "prediction-datetime"
  ).value = `${year}-${month}-${day}T${hours}:${minutes}`;

  // Add event listeners for filters
  document
    .getElementById("city-select")
    .addEventListener("change", async event => {
      const selectedCity = event.target.value;
      await populateRouteAndZoneFilters(selectedCity);
      // No auto-fetch here, user clicks "Get Predictions"
    });

  document
    .getElementById("apply-filters-btn")
    .addEventListener("click", fetchAndRenderPredictions);

  // Initial fetch of predictions based on default selections
  // Only if city, route, zone are actually pre-selected and valid
  if (initialCitySelect.value) {
    await fetchAndRenderPredictions();
  }

  // Modal close functionality
  const modal = document.getElementById("detail-chart-modal");
  const closeButton = document.querySelector(".close-button");

  closeButton.onclick = () => {
    modal.style.display = "none";
    if (modalChartInstance) {
      modalChartInstance.destroy();
      modalChartInstance = null;
    }
  };

  window.onclick = event => {
    if (event.target == modal) {
      modal.style.display = "none";
      if (modalChartInstance) {
        modalChartInstance.destroy();
        modalChartInstance = null;
      }
    }
  };
});

async function fetchData(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const errorMessage =
        errorData.error || `HTTP error! status: ${response.status}`;
      // Log the full response text if it's not JSON, for debugging
      if (response.headers.get("content-type")?.includes("text/html")) {
        const text = await response.text();
        console.error("Non-JSON error response from server:", text);
      }
      throw new Error(errorMessage);
    }
    return await response.json();
  } catch (error) {
    console.error("Error fetching data:", error);
    // Use a more subtle notification for fetching errors, not blocking alerts everywhere
    // For predictions specifically, render a message in the container
    return { error: `Failed to fetch data: ${error.message}` };
  }
}

async function fetchOverallMetrics() {
  const data = await fetchData("/metrics");
  if (data && !data.error && data.server_metrics) {
    const serverMetrics = data.server_metrics;
    const clientMetrics = data.client_metrics; // Get client metrics

    // Display latest round metrics in cards
    if (serverMetrics.length > 0) {
      const latestServerMetric = serverMetrics[serverMetrics.length - 1];
      document.getElementById("latest-round-display").textContent =
        latestServerMetric.round;

      // Calculate Avg Client Fit Loss from client_metrics for the latest round
      const latestRoundClientMetrics = clientMetrics.filter(
        m =>
          m.round === latestServerMetric.round &&
          m.fit_loss !== null &&
          !isNaN(m.fit_loss)
      );
      const totalFitLoss = latestRoundClientMetrics.reduce(
        (sum, m) => sum + m.fit_loss,
        0
      );
      const avgFitLoss =
        latestRoundClientMetrics.length > 0
          ? totalFitLoss / latestRoundClientMetrics.length
          : null;

      document.getElementById("avg-fit-loss-display").textContent =
        avgFitLoss !== null ? avgFitLoss.toFixed(4) : "N/A";

      // Display Avg Eval Loss (from server_metrics)
      document.getElementById("avg-eval-loss-display").textContent =
        latestServerMetric.avg_client_loss // This is now avg_eval_loss from server_metrics
          ? latestServerMetric.avg_client_loss.toFixed(4)
          : "N/A";

      document.getElementById("avg-mae-display").textContent =
        latestServerMetric.avg_mae
          ? latestServerMetric.avg_mae.toFixed(4)
          : "N/A";
      const maxEpsilonDisplay = document.getElementById("max-epsilon-display");
      if (maxEpsilonDisplay) {
        maxEpsilonDisplay.textContent = latestServerMetric.max_epsilon
          ? latestServerMetric.max_epsilon.toFixed(2)
          : "N/A";
      }
    }

    // Prepare data for overall metrics chart
    const rounds = serverMetrics.map(m => m.round).sort((a, b) => a - b);
    const avgEvalLosses = serverMetrics.map(m => m.avg_client_loss); // Renamed from avg_client_loss
    const avgMaes = serverMetrics.map(m => m.avg_mae);
    const avgEpsilons = serverMetrics.map(m => m.avg_epsilon);

    const ctx = document.getElementById("overallMetricsChart").getContext("2d");
    if (overallMetricsChart) {
      overallMetricsChart.destroy(); // Destroy old chart instance
    }
    overallMetricsChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: rounds,
        datasets: [
          {
            label: "Avg Eval Loss", // Updated label
            data: avgEvalLosses,
            borderColor: "rgb(75, 192, 192)",
            tension: 0.1,
            yAxisID: "y",
            pointRadius: 3,
            pointHoverRadius: 5,
          },
          {
            label: "Avg MAE (Eval)",
            data: avgMaes,
            borderColor: "rgb(255, 99, 132)",
            tension: 0.1,
            yAxisID: "y",
            pointRadius: 3,
            pointHoverRadius: 5,
          },
          {
            label: "Avg Epsilon",
            data: avgEpsilons,
            borderColor: "rgb(54, 162, 235)",
            tension: 0.1,
            hidden: true, // Hide by default
            yAxisID: "y1", // Use a separate Y-axis for epsilon if values differ greatly
            pointRadius: 3,
            pointHoverRadius: 5,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            type: "category",
            title: {
              display: true,
              text: "Round",
            },
            grid: {
              display: false,
            },
          },
          y: {
            type: "linear",
            display: true,
            position: "left",
            title: {
              display: true,
              text: "Loss / MAE",
            },
            grid: {
              color: "rgba(0, 0, 0, 0.05)",
            },
          },
          y1: {
            type: "linear",
            display: true,
            position: "right",
            title: {
              display: true,
              text: "Epsilon",
            },
            grid: {
              drawOnChartArea: false, // Only draw grid lines for the first Y-axis
            },
          },
        },
        plugins: {
          tooltip: {
            mode: "index",
            intersect: false,
            backgroundColor: "rgba(0, 0, 0, 0.7)",
            titleFont: { size: 14 },
            bodyFont: { size: 12 },
            padding: 10,
            cornerRadius: 6,
          },
          legend: {
            labels: {
              font: {
                size: 12,
              },
            },
          },
        },
      },
    });
  } else if (data && data.error) {
    console.error("Error fetching overall metrics:", data.error);
    // You might want to display a message on the dashboard if metrics can't be loaded
  }
}

async function populateRouteAndZoneFilters(city) {
  const routeSelect = document.getElementById("route-select");
  const zoneSelect = document.getElementById("zone-select");

  routeSelect.innerHTML = '<option value="all">All Routes</option>';
  zoneSelect.innerHTML = '<option value="all">All Zones</option>';

  if (!city) {
    return; // No city selected, no routes/zones to load
  }

  // Fetch routes (always using the latest round implicitly on backend)
  const routesData = await fetchData(`/routes_for_city?city=${city}`);
  if (routesData && !routesData.error) {
    routesData.forEach(route => {
      const option = document.createElement("option");
      option.value = route;
      option.textContent = route;
      routeSelect.appendChild(option);
    });
  } else if (routesData && routesData.error) {
    console.error("Error fetching routes:", routesData.error);
    // Optionally, display a message to the user
  }

  // Fetch zones (always using the latest round implicitly on backend)
  const zonesData = await fetchData(`/zones_for_city?city=${city}`);
  if (zonesData && !zonesData.error) {
    zonesData.forEach(zone => {
      const option = document.createElement("option");
      option.value = zone;
      option.textContent = zone;
      zoneSelect.appendChild(option);
    });
  } else if (zonesData && zonesData.error) {
    console.error("Error fetching zones:", zonesData.error);
    // Optionally, display a message to the user
  }
}

async function fetchAndRenderPredictions() {
  const city = document.getElementById("city-select").value;
  const route_id = document.getElementById("route-select").value;
  const zone = document.getElementById("zone-select").value;
  const datetime = document.getElementById("prediction-datetime").value; // Get value from new input

  const resultsContainer = document.getElementById(
    "prediction-results-container"
  );
  const summaryCard = document.getElementById("prediction-summary-card");
  const applyButton = document.getElementById("apply-filters-btn");

  resultsContainer.innerHTML =
    '<p class="initial-message">Loading predictions...</p>';
  summaryCard.classList.add("hidden"); // Hide summary while loading
  applyButton.disabled = true; // Disable button during fetch
  applyButton.textContent = "Loading...";

  if (!city || !datetime) {
    resultsContainer.innerHTML =
      '<p class="initial-message error-message">Please select a city and a prediction date & time.</p>';
    applyButton.disabled = false;
    applyButton.textContent = "Get Predictions";
    return;
  }

  let url = `/predictions_data?city=${city}&datetime=${datetime}`;
  if (route_id && route_id !== "all") {
    url += `&route_id=${route_id}`;
  }
  if (zone && zone !== "all") {
    url += `&zone=${zone}`;
  }

  const data = await fetchData(url);

  // Re-enable button and reset text
  applyButton.disabled = false;
  applyButton.textContent = "Get Predictions";

  renderPredictionResults(data);
}

function renderPredictionResults(chartData) {
  const container = document.getElementById("prediction-results-container");
  const summaryCard = document.getElementById("prediction-summary-card");
  container.innerHTML = ""; // Clear previous results

  // Handle potential error messages from the backend
  if (chartData && chartData.error) {
    container.innerHTML = `<p class="initial-message error-message">Error: ${chartData.error}</p>`;
    summaryCard.classList.add("hidden"); // Ensure summary card is hidden on error
    return;
  }

  if (!chartData || Object.keys(chartData).length === 0) {
    container.innerHTML =
      '<p class="initial-message">No prediction data available for the selected filters.</p>';
    summaryCard.classList.add("hidden"); // Ensure summary card is hidden on no data
    return;
  }

  let overallPredicted = 0;
  let overallActual = 0;
  let totalMAE = 0;
  let predictionCount = 0;
  let actualCount = 0;
  let busyCount = 0;

  for (const comboKey in chartData) {
    const data = chartData[comboKey];
    const routeId = data.route_id;
    const zone = data.zone;
    const isBusy = data.is_busy; // Get the busy flag
    const predictedVal = data.predicted[0]; // Single predicted value
    const actualVal = data.actual[0]; // Single actual value
    const mae = data.mae; // Single MAE value from backend
    const accuracyPercentage = data.accuracy_percentage; // Single accuracy from backend

    overallPredicted += predictedVal;
    predictionCount++;

    if (actualVal !== null && typeof actualVal !== "undefined") {
      overallActual += actualVal;
      actualCount++;
    }
    if (mae !== null && typeof mae !== "undefined") {
      totalMAE += mae;
    }
    if (isBusy) {
      busyCount++;
    }

    let accuracyClass = "accuracy-good";
    if (
      accuracyPercentage === "N/A" ||
      (typeof accuracyPercentage === "string" &&
        accuracyPercentage.includes("NaN")) ||
      (parseFloat(accuracyPercentage) < 70 && actualVal !== null)
    ) {
      accuracyClass = "accuracy-poor";
    } else if (parseFloat(accuracyPercentage) < 85) {
      accuracyClass = "accuracy-moderate";
    }

    let busyClass = isBusy ? "busy-card" : "";

    // Create a card for each route's prediction summary
    const routeCard = document.createElement("div");
    routeCard.className = `route-card ${busyClass}`; // Add busy class
    routeCard.innerHTML = `
            <h4>Route: ${routeId} / Zone: ${zone}</h4>
            <div class="route-stats">
                <div class="stat-item">
                    <span class="stat-value">${predictedVal.toFixed(0)}</span>
                    <span class="stat-label">Predicted</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">${
                      actualVal !== null ? actualVal.toFixed(0) : "N/A"
                    }</span>
                    <span class="stat-label">Actual</span>
                </div>
            </div>
            <div class="accuracy-indicator ${accuracyClass}">
                Accuracy: ${accuracyPercentage}%
            </div>
            ${isBusy ? '<div class="busy-indicator">🚨 Busy!</div>' : ""}
        `;
    routeCard.addEventListener("click", () =>
      showDetailedChart(routeId, zone, data)
    ); // Pass the entire data object
    container.appendChild(routeCard);
  }

  // Update Overall Summary Card
  const overallAverageMAE = actualCount > 0 ? totalMAE / actualCount : "N/A";
  const overallAccuracy =
    overallActual > 0
      ? ((1 - overallAverageMAE / (overallActual / actualCount)) * 100).toFixed(
          1
        )
      : "N/A";
  document.getElementById("overall-predicted").textContent =
    overallPredicted.toFixed(0);
  document.getElementById("overall-actual").textContent =
    overallActual > 0 ? overallActual.toFixed(0) : "N/A";
  document.getElementById(
    "overall-accuracy"
  ).textContent = `${overallAccuracy}%`;
  document.getElementById("busy-count").textContent = `${busyCount} out of ${
    Object.keys(chartData).length
  }`;
  summaryCard.classList.remove("hidden"); // Show summary card
}

function showDetailedChart(routeId, zone, data) {
  const modal = document.getElementById("detail-chart-modal");
  const modalChartCanvas = document.getElementById("modalChart");
  const modalChartTitle = document.getElementById("modal-chart-title");

  modalChartTitle.textContent = `Detailed Predictions for Route: ${routeId} / Zone: ${zone}`;

  // Update modal metrics display with values from the passed data object
  document.getElementById("modal-predicted-value").textContent =
    data.predicted[0].toFixed(0);
  document.getElementById("modal-actual-value").textContent =
    data.actual[0] !== null ? data.actual[0].toFixed(0) : "N/A";
  document.getElementById("modal-mae-value").textContent =
    data.mae !== null ? data.mae.toFixed(2) : "N/A";
  document.getElementById("modal-accuracy-value").textContent =
    data.accuracy_percentage !== "N/A" ? `${data.accuracy_percentage}%` : "N/A";

  if (modalChartInstance) {
    modalChartInstance.destroy(); // Destroy previous chart instance
  }

  modalChartInstance = new Chart(modalChartCanvas.getContext("2d"), {
    type: "line",
    data: {
      labels: data.labels, // This will be a single datetime string
      datasets: [
        {
          label: "Actual",
          data: data.actual, // This will be a single actual value or null
          borderColor: "rgb(75, 192, 192)",
          tension: 0.1,
          pointRadius: 6, // Larger point for single data point
          pointHoverRadius: 8,
          // If all actual values are null/undefined, hide this dataset
          hidden: data.actual.every(
            val => val === null || typeof val === "undefined"
          ),
        },
        {
          label: "Predicted",
          data: data.predicted, // This will be a single predicted value
          borderColor: "rgb(255, 99, 132)",
          tension: 0.1,
          pointRadius: 6, // Larger point for single data point
          pointHoverRadius: 8,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          type: "time",
          time: {
            unit: "hour", // Still use hour for display, even if single point
            tooltipFormat: "yyyy-MM-dd HH:mm",
            displayFormats: {
              hour: "MMM dd HH:mm",
            },
          },
          title: {
            display: true,
            text: "Date and Time",
          },
          grid: {
            display: false,
          },
        },
        y: {
          title: {
            display: true,
            text: "Passenger Count",
          },
          grid: {
            color: "rgba(0, 0, 0, 0.05)",
          },
        },
      },
      plugins: {
        tooltip: {
          mode: "index",
          intersect: false,
          backgroundColor: "rgba(0, 0, 0, 0.7)",
          titleFont: { size: 14 },
          bodyFont: { size: 12 },
          padding: 10,
          cornerRadius: 6,
        },
        legend: {
          labels: {
            font: {
              size: 12,
            },
          },
        },
      },
    },
  });

  modal.style.display = "flex"; // Show the modal
}
