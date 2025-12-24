import { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';

export const ModelAnswersAccordion = ({ results }) => {
  const [expandedModels, setExpandedModels] = useState(new Set());

  if (!results || !Array.isArray(results) || results.length === 0) {
    return null;
  }

  // Group results by model
  const resultsByModel = {};
  results.forEach(result => {
    if (!resultsByModel[result.model]) {
      resultsByModel[result.model] = [];
    }
    resultsByModel[result.model].push(result);
  });

  // Calculate fastest correct answers count per model
  const calculateFastestCorrectCount = (modelResults, modelName) => {
    // Group all results by file to find fastest correct answer per file
    const fastestByFile = {};
    results.forEach(result => {
      if (result.correct) {
        if (!fastestByFile[result.file] || result.response_time < fastestByFile[result.file].response_time) {
          fastestByFile[result.file] = result;
        }
      }
    });
    
    // Count how many files where this model is the fastest correct
    let fastestCorrectCount = 0;
    Object.values(fastestByFile).forEach(fastest => {
      if (fastest.model === modelName) {
        fastestCorrectCount++;
      }
    });
    
    return fastestCorrectCount;
  };

  const toggleModel = (modelName) => {
    const newExpanded = new Set(expandedModels);
    if (newExpanded.has(modelName)) {
      newExpanded.delete(modelName);
    } else {
      newExpanded.add(modelName);
    }
    setExpandedModels(newExpanded);
  };

  return (
    <div className="card">
      <div className="p-6">
        <div className="flex items-center space-x-3 mb-4">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
            Model Answers Details
          </h2>
          {results[0]?.evaluator_model && (
            <span className="text-sm text-gray-500 dark:text-gray-400">
              (evaluated by {results[0].evaluator_model})
            </span>
          )}
        </div>
        <div className="space-y-4">
          {Object.entries(resultsByModel).sort(([a], [b]) => a.localeCompare(b)).map(([modelName, modelResults]) => (
            <div key={modelName} className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
              <button
                onClick={() => toggleModel(modelName)}
                className="w-full px-4 py-3 bg-gray-50 dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors flex items-center justify-between text-left"
              >
                <div className="flex items-center space-x-3">
                  {expandedModels.has(modelName) ? (
                    <ChevronDown className="w-5 h-5 text-gray-500 dark:text-gray-400" />
                  ) : (
                    <ChevronRight className="w-5 h-5 text-gray-500 dark:text-gray-400" />
                  )}
                  <span className="font-medium text-gray-900 dark:text-white">
                    {modelName}
                  </span>
                  <span className="text-sm text-gray-500 dark:text-gray-400">
                    ({modelResults.length} answers)
                  </span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-sm text-green-600 dark:text-green-400">
                    {modelResults.filter(r => r.correct).length} correct
                  </span>
                  <span className="text-sm text-yellow-600 dark:text-yellow-400">
                    / {calculateFastestCorrectCount(modelResults, modelName)} ✦
                  </span>
                  <span className="text-sm text-red-600 dark:text-red-400">
                    / {modelResults.filter(r => !r.correct).length} incorrect
                  </span>
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    ({modelResults.reduce((sum, r) => sum + (typeof r.response_time === 'number' ? r.response_time : 0), 0).toFixed(1)}s)
                  </span>
                  <span className="text-sm text-blue-600 dark:text-blue-400">
                    [{(modelResults.filter(r => r.correct).length * 100 / modelResults.reduce((sum, r) => sum + (typeof r.response_time === 'number' ? r.response_time : 0), 0)).toFixed(2)}]
                  </span>
                </div>
              </button>
              
              {expandedModels.has(modelName) && (
                <div className="border-t border-gray-200 dark:border-gray-700">
                  <div className="divide-y divide-gray-200 dark:divide-gray-700">
                    {modelResults.map((result, index) => (
                      <div key={index} className="p-4">
                        <div className="flex items-start justify-between mb-3">
                          <div className="flex-1">
                            <div className="flex items-center space-x-2 mb-2">
                              <span className="font-medium text-gray-900 dark:text-white">
                                {result.file}
                              </span>
                              <span className={`px-2 py-1 rounded text-xs font-medium ${
                                result.correct 
                                  ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300'
                                  : 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
                              }`}>
                                {result.correct ? 'Correct' : 'Incorrect'}
                              </span>
                              <span className="text-sm text-gray-500 dark:text-gray-400">
                                {typeof result.response_time === 'number' ? result.response_time.toFixed(2) + 's' : 'N/A'}
                              </span>
                            </div>
                            
                            <div className="space-y-3">
                              <div>
                                <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-1">
                                  Prompt:
                                </h4>
                                <pre className="text-xs bg-gray-100 dark:bg-gray-800 p-2 rounded text-gray-600 dark:text-gray-400 whitespace-pre-wrap max-h-32 overflow-y-auto">
                                  {result.prompt}
                                </pre>
                              </div>
                              
                              <div>
                                <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-1">
                                  Expected:
                                </h4>
                                <pre className="text-xs bg-gray-100 dark:bg-gray-800 p-2 rounded text-gray-600 dark:text-gray-400 whitespace-pre-wrap max-h-32 overflow-y-auto">
                                  {result.expected}
                                </pre>
                              </div>
                              
                              <div>
                                <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-1">
                                  Generated:
                                </h4>
                                <pre className="text-xs bg-gray-100 dark:bg-gray-800 p-2 rounded text-gray-600 dark:text-gray-400 whitespace-pre-wrap max-h-32 overflow-y-auto">
                                  {result.generated}
                                </pre>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};