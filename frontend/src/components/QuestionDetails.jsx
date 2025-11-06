import { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { groupResultsByFile } from '../utils/dataProcessing';

export const QuestionDetails = ({ results }) => {
  const [expandedFiles, setExpandedFiles] = useState(new Set());
  const resultsByFile = groupResultsByFile(results);

  const toggleFile = (filename) => {
    const newExpanded = new Set(expandedFiles);
    if (newExpanded.has(filename)) {
      newExpanded.delete(filename);
    } else {
      newExpanded.add(filename);
    }
    setExpandedFiles(newExpanded);
  };

  return (
    <div className="card">
      <div className="p-6">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          Question Details
        </h2>
        <div className="space-y-4">
          {Object.entries(resultsByFile).map(([file, data]) => (
            <div 
              key={file} 
              className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden"
            >
              <button
                onClick={() => toggleFile(file)}
                className="w-full px-4 py-3 bg-gray-50 dark:bg-gray-700/50 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors flex items-center justify-between text-left"
              >
                <span className="font-medium text-gray-900 dark:text-white">
                  {file}
                </span>
                {expandedFiles.has(file) ? (
                  <ChevronDown className="w-4 h-4 text-gray-500" />
                ) : (
                  <ChevronRight className="w-4 h-4 text-gray-500" />
                )}
              </button>
              
              {expandedFiles.has(file) && (
                <div className="p-4 space-y-4 animate-fade-in">
                  <div>
                    <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                      Prompt:
                    </h4>
                    <pre className="bg-gray-100 dark:bg-gray-800 p-3 rounded text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap">
                      {data.prompt}
                    </pre>
                  </div>
                  
                  <div>
                    <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                      Expected Answer:
                    </h4>
                    <pre className="bg-gray-100 dark:bg-gray-800 p-3 rounded text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap">
                      {data.expected}
                    </pre>
                  </div>
                  
                  <hr className="border-gray-200 dark:border-gray-700" />
                  
                  <div className="space-y-3">
                    {data.models.map((modelResult, index) => (
                      <div 
                        key={index}
                        className={`p-4 rounded-lg border-l-4 ${
                          modelResult.correct 
                            ? 'bg-green-50 dark:bg-green-900/20 border-green-500' 
                            : 'bg-red-50 dark:bg-red-900/20 border-red-500'
                        }`}
                      >
                        <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                          {modelResult.model}
                        </h4>
                        <div className="mb-2">
                          <span className="text-sm font-medium text-gray-600 dark:text-gray-400">
                            Generated Answer:
                          </span>
                          <pre className="mt-1 bg-white dark:bg-gray-800 p-3 rounded text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap">
                            {modelResult.generated}
                          </pre>
                        </div>
                        <div className="text-sm text-gray-600 dark:text-gray-400">
                          <em>Response Time: {modelResult.response_time.toFixed(2)}s</em>
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