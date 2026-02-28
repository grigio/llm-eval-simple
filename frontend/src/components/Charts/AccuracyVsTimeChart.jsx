import { useState, useCallback, useMemo } from 'react';
import { Scatter } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(LinearScale, PointElement, LineElement, Tooltip, Legend);

const COLORS = [
  'rgba(59, 130, 246, 1)',
  'rgba(239, 68, 68, 1)',
  'rgba(34, 197, 94, 1)',
  'rgba(168, 85, 247, 1)',
  'rgba(249, 115, 22, 1)',
  'rgba(236, 72, 153, 1)',
  'rgba(20, 184, 166, 1)',
  'rgba(132, 204, 22, 1)',
];

const getColor = (index) => COLORS[index % COLORS.length];

const afterDatasetsDraw = (chart) => {
  const ctx = chart.ctx;
  const meta = chart.getDatasetMeta(0);
  const dataPoints = chart.data.datasets[0].data;
  
  meta.data.forEach((point, index) => {
    const dataPoint = dataPoints[index];
    if (dataPoint?.selected) {
      const { x, y } = point.getProps(['x', 'y']);
      const colorIndex = dataPoint.colorIndex ?? index;
      
      ctx.save();
      ctx.font = 'bold 12px system-ui';
      ctx.fillStyle = getColor(colorIndex);
      ctx.textAlign = 'left';
      ctx.fillText(dataPoint.model, x + 12, y + 4);
      ctx.restore();
    }
  });
};

export const AccuracyVsTimeChart = ({ results }) => {
  const [selectedModels, setSelectedModels] = useState([]);

  if (!results || !Array.isArray(results) || results.length === 0) {
    return null;
  }

  const modelStats = {};
  results.forEach(result => {
    if (!result.model || typeof result.response_time !== 'number') return;
    
    if (!modelStats[result.model]) {
      modelStats[result.model] = { correct: 0, total: 0, times: [] };
    }
    modelStats[result.model].total += 1;
    if (result.correct) {
      modelStats[result.model].correct += 1;
    }
    modelStats[result.model].times.push(result.response_time);
  });

  const modelList = Object.keys(modelStats);
  const dataPoints = modelList.map((model, index) => {
    const stats = modelStats[model];
    const avgTime = stats.times.reduce((a, b) => a + b, 0) / stats.times.length;
    const accuracy = (stats.correct / stats.total) * 100;
    return {
      x: parseFloat(avgTime.toFixed(2)),
      y: parseFloat(accuracy.toFixed(1)),
      model,
      colorIndex: index,
    };
  });

  const handlePointClick = useCallback((event, elements) => {
    if (elements.length > 0) {
      const index = elements[0].index;
      const model = dataPoints[index].model;
      setSelectedModels(prev => 
        prev.includes(model) 
          ? prev.filter(m => m !== model)
          : [...prev, model]
      );
    }
  }, [dataPoints]);

  const dataPointsWithSelection = useMemo(() => 
    dataPoints.map(p => ({
      ...p,
      selected: selectedModels.includes(p.model)
    }))
  , [dataPoints, selectedModels]);

  const chartData = useMemo(() => ({
    datasets: [
      {
        label: 'Models',
        data: dataPointsWithSelection,
        backgroundColor: dataPointsWithSelection.map(p => 
          p.selected 
            ? getColor(p.colorIndex) 
            : 'rgba(156, 163, 175, 0.5)'
        ),
        borderColor: dataPointsWithSelection.map(p => getColor(p.colorIndex)),
        borderWidth: 2,
        pointRadius: dataPointsWithSelection.map(p => p.selected ? 10 : 6),
        pointHoverRadius: 10,
      },
    ],
  }), [dataPointsWithSelection]);

  const chartOptions = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    onClick: handlePointClick,
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: true,
        text: 'Accuracy vs Response Time (click points to show labels)',
        color: 'rgb(55, 65, 81)',
        font: {
          size: 16,
          weight: 'bold',
        },
      },
      tooltip: {
        callbacks: {
          label: (context) => {
            const point = context.raw;
            return `${point.model}: ${point.y}% accuracy, ${point.x}s avg time`;
          },
        },
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Avg Response Time (s)',
          color: 'rgb(55, 65, 81)',
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)',
        },
      },
      y: {
        title: {
          display: true,
          text: 'Accuracy (%)',
          color: 'rgb(55, 65, 81)',
        },
        beginAtZero: true,
        max: 100,
        ticks: {
          callback: (value) => `${value}%`,
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)',
        },
      },
    },
  }), [handlePointClick]);

  return (
    <div className="card p-6">
      <div className="flex justify-between items-center mb-4">
        <div className="h-80 flex-1">
          <Scatter key={selectedModels.join(',')} data={chartData} options={chartOptions} plugins={[{
            id: 'afterDatasetsDraw',
            afterDatasetsDraw
          }]} />
        </div>
        {selectedModels.length > 0 && (
          <button
            onClick={() => setSelectedModels([])}
            className="ml-4 px-3 py-1 text-sm bg-gray-200 dark:bg-gray-700 rounded hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
          >
            Clear
          </button>
        )}
      </div>
    </div>
  );
};
