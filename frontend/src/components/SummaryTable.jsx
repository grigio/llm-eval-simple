import { calculateModelSummary, formatAccuracy, formatResponseTime } from '../utils/dataProcessing';

export const SummaryTable = ({ results }) => {
  const modelSummary = calculateModelSummary(results);

  return (
    <div className="card overflow-hidden">
      <div className="p-6">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          Model Performance Summary
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-700">
                <th className="text-left py-3 px-4 font-semibold text-gray-900 dark:text-white">
                  Model
                </th>
                <th className="text-left py-3 px-4 font-semibold text-gray-900 dark:text-white">
                  Correct
                </th>
                <th className="text-left py-3 px-4 font-semibold text-gray-900 dark:text-white">
                  Avg Response Time
                </th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(modelSummary).map(([model, stats]) => {
                const total = stats.total;
                const correct = stats.correct;
                const totalTime = stats.totalTime;
                const accuracy = total > 0 ? (correct / total) * 100 : 0;
                const avgTime = total > 0 ? totalTime / total : 0;

                return (
                  <tr 
                    key={model} 
                    className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
                  >
                    <td className="py-3 px-4">
                      <span className="font-medium text-gray-900 dark:text-white">
                        {model}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center space-x-3">
                        <span className="text-sm text-gray-600 dark:text-gray-400">
                          {correct}/{total} ({formatAccuracy(correct, total)})
                        </span>
                        <div className="flex-1 max-w-xs">
                          <div className="progress-bar">
                            <div 
                              className="progress-fill"
                              style={{ width: `${accuracy}%` }}
                            />
                          </div>
                        </div>
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        {formatResponseTime(avgTime)}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};