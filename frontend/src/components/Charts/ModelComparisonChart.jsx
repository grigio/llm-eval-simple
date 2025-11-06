import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';
import { getUniquePromptsAndModels } from '../../utils/dataProcessing';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

export const ModelComparisonChart = ({ results }) => {
  const { prompts, models } = getUniquePromptsAndModels(results);
  
  const datasets = models.map((model, index) => {
    const colors = [
      'rgb(34, 197, 94)',
      'rgb(59, 130, 246)', 
      'rgb(239, 68, 68)',
      'rgb(245, 158, 11)',
      'rgb(139, 92, 246)',
      'rgb(236, 72, 153)',
    ];
    
    const modelResults = prompts.map(prompt => {
      const result = results.find(r => r.model === model && r.file === prompt);
      return result ? (result.correct ? 1 : 0) : null;
    });

    return {
      label: model,
      data: modelResults,
      borderColor: colors[index % colors.length],
      backgroundColor: colors[index % colors.length] + '20',
      tension: 0.1,
      pointRadius: 4,
      pointHoverRadius: 6,
    };
  });

  const chartData = {
    labels: prompts,
    datasets,
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Model Performance Across Prompts',
        color: 'rgb(55, 65, 81)',
        font: {
          size: 16,
          weight: 'bold',
        },
      },
      tooltip: {
        callbacks: {
          label: (context) => {
            const value = context.parsed.y;
            return `${context.dataset.label}: ${value === 1 ? 'Correct' : value === 0 ? 'Incorrect' : 'No data'}`;
          },
        },
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 1.2,
        ticks: {
          stepSize: 1,
          callback: (value) => {
            if (value === 1) return 'Correct';
            if (value === 0) return 'Incorrect';
            return '';
          },
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)',
        },
      },
      x: {
        grid: {
          display: false,
        },
        ticks: {
          maxRotation: 45,
          minRotation: 45,
        },
      },
    },
  };

  return (
    <div className="card p-6">
      <div className="h-80">
        <Line data={chartData} options={options} />
      </div>
    </div>
  );
};