import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';
import { calculateModelSummary } from '../../utils/dataProcessing';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

export const ResponseTimeChart = ({ results }) => {
  const modelSummary = calculateModelSummary(results);
  
  const chartData = {
    labels: Object.keys(modelSummary),
    datasets: [
      {
        label: 'Avg Response Time (s)',
        data: Object.values(modelSummary).map(stats => {
          const avgTime = stats.total > 0 ? stats.totalTime / stats.total : 0;
          return parseFloat(avgTime.toFixed(2));
        }),
        backgroundColor: 'rgba(59, 130, 246, 0.8)',
        borderColor: 'rgba(59, 130, 246, 1)',
        borderWidth: 1,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: true,
        text: 'Average Response Time Comparison',
        color: 'rgb(55, 65, 81)',
        font: {
          size: 16,
          weight: 'bold',
        },
      },
      tooltip: {
        callbacks: {
          label: (context) => {
            return `Avg Time: ${context.parsed.y}s`;
          },
        },
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          callback: (value) => `${value}s`,
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)',
        },
      },
      x: {
        grid: {
          display: false,
        },
      },
    },
  };

  return (
    <div className="card p-6">
      <div className="h-80">
        <Bar data={chartData} options={options} />
      </div>
    </div>
  );
};