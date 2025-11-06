import { useState, useEffect } from 'react';
import { useApi, useJsonFile } from '../hooks/useApi';
import { Loader2 } from 'lucide-react';
import { ResultsMatrix } from './ResultsMatrix';
import { ModelAnswersAccordion } from './ModelAnswersAccordion';
import { DarkModeToggle } from './ui/DarkModeToggle';

export const Dashboard = () => {
  const [renderFile, setRenderFile] = useState(null);
  
  // Parse URL parameters for render query
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const renderParam = params.get('render');
    
    if (renderParam) {
      setRenderFile(renderParam);
    } else if (window.location.pathname === '/' && !window.location.search) {
      // Redirect to default JSON file if no parameters
      window.location.href = '/?render=report-evaluated.json';
    }
  }, []);

  const { data: apiData, loading: apiLoading, error: apiError } = useApi('/api/results');
  const { data: jsonData, loading: jsonLoading, error: jsonError } = useJsonFile(renderFile);
  
  const results = renderFile ? jsonData : (apiData?.results || []);
  const loading = renderFile ? jsonLoading : apiLoading;
  const error = renderFile ? jsonError : apiError;

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4 text-primary-600" />
          <p className="text-gray-600 dark:text-gray-400">Loading evaluation results...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
        <div className="text-center max-w-md">
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6">
            <h2 className="text-xl font-semibold text-red-800 dark:text-red-300 mb-2">
              Error Loading Results
            </h2>
            <p className="text-red-600 dark:text-red-400">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  if (!results || results.length === 0) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
            No Results Available
          </h2>
          <p className="text-gray-600 dark:text-gray-400">
            Please run the evaluation first to generate results.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div>
              <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                LLM Eval Dashboard
              </h1>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Interactive model evaluation results
              </p>
            </div>
            <DarkModeToggle />
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="space-y-8">
          {renderFile && (
            <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
              <p className="text-blue-800 dark:text-blue-300">
                Rendering heatmap for: <span className="font-mono font-semibold">{renderFile}</span>
              </p>
            </div>
          )}
          
          {/* Heatmap Results Display */}
          <ResultsMatrix results={results} />
          
          {/* Model Answers Accordion */}
          <ModelAnswersAccordion results={results} />
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-sm text-gray-600 dark:text-gray-400">
            LLM Eval Simple - Interactive Dashboard
          </p>
        </div>
      </footer>
    </div>
  );
};