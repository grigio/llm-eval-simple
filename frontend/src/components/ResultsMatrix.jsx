import { useState } from 'react';
import { Modal } from './ui/Modal';
import { getHeatmapColor, getResponseTimeStats } from '../utils/heatmapUtils';
import { useDarkMode } from '../hooks/useDarkMode';

export const ResultsMatrix = ({ results }) => {
  const [selectedCell, setSelectedCell] = useState(null);
  const { isDark } = useDarkMode();
  
  // Add safety checks
  if (!results || !Array.isArray(results) || results.length === 0) {
    return (
      <div className="card">
        <div className="p-6">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            Evaluation Heatmap
          </h2>
          <p className="text-gray-600 dark:text-gray-400">
            No results available to display.
          </p>
        </div>
      </div>
    );
  }

  // Get unique models and files safely
  const models = [...new Set(results.map(r => r?.model).filter(Boolean))].sort();
  const prompts = [...new Set(results.map(r => r?.file).filter(Boolean))].sort();
  
  // Calculate response time stats for color normalization
  const { min: minTime, max: maxTime } = getResponseTimeStats(results);

  // Find the fastest correct result for each test (file)
  const fastestCorrectByFile = {};
  results.forEach(result => {
    if (result.correct && typeof result.response_time === 'number') {
      const file = result.file;
      if (!fastestCorrectByFile[file] || result.response_time < fastestCorrectByFile[file].response_time) {
        fastestCorrectByFile[file] = result;
      }
    }
  });

  const handleCellClick = (model, file) => {
    const result = results.find(r => r.model === model && r.file === file);
    if (result) {
      setSelectedCell({
        model: result.model,
        file: result.file,
        prompt: result.prompt,
        generated: result.generated,
        expected: result.expected,
        response_time: result.response_time?.toFixed(2) + 's',
        correct: result.correct,
        evaluator_model: result.evaluator_model,
        note: result.note || ''
      });
    }
  };

  const navigateToModel = (direction) => {
    if (!selectedCell) return;
    
    const currentIndex = models.indexOf(selectedCell.model);
    let newIndex;
    
    if (direction === 'prev') {
      newIndex = currentIndex > 0 ? currentIndex - 1 : models.length - 1;
    } else {
      newIndex = currentIndex < models.length - 1 ? currentIndex + 1 : 0;
    }
    
    const nextModel = models[newIndex];
    const result = results.find(r => r.model === nextModel && r.file === selectedCell.file);
    
    if (result) {
      setSelectedCell({
        model: result.model,
        file: result.file,
        prompt: result.prompt,
        generated: result.generated,
        expected: result.expected,
        response_time: result.response_time?.toFixed(2) + 's',
        correct: result.correct,
        evaluator_model: result.evaluator_model,
        note: result.note || ''
      });
    }
  };

  const getHeatmapCell = (model, prompt) => {
    const result = results.find(r => r.model === model && r.file === prompt);
    
    if (!result || typeof result.response_time !== 'number') {
      return (
        <td
          key={`${model}-${prompt}`}
          className="table-cell bg-gray-100 dark:bg-gray-700 text-gray-400 dark:text-gray-500 text-center"
        >
          -
        </td>
      );
    }
    
    const colors = getHeatmapColor(result.correct, result.response_time, minTime, maxTime, isDark);
    const isFastestCorrect = fastestCorrectByFile[prompt] === result;
    
    return (
      <td
        key={`${model}-${prompt}`}
        className="table-cell cursor-pointer transition-all duration-300 hover:scale-105 hover:shadow-lg relative group p-0"
        style={{ 
          backgroundColor: colors.bg,
          color: colors.text
        }}
        onClick={() => handleCellClick(model, prompt)}
        onMouseEnter={(e) => {
          e.currentTarget.style.backgroundColor = colors.hover;
          e.currentTarget.style.transform = 'scale(1.05)';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.backgroundColor = colors.bg;
          e.currentTarget.style.transform = 'scale(1)';
        }}
      >
        <div className="text-center font-bold text-sm p-2 whitespace-nowrap">
          {isFastestCorrect && (
            <span className="mr-1 text-yellow-500 text-lg" title="Fastest correct result">✦</span>
          )}
          {result.response_time.toFixed(2)}s
        </div>
        <div className="absolute bottom-0 left-0 right-0 h-1 opacity-50" 
             style={{ backgroundColor: colors.intensity === 'high' ? colors.text : 'transparent' }} />
      </td>
    );
  };

  return (
    <>
      <div className="card overflow-hidden">
        <div className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
              Evaluation Heatmap
            </h2>
            <div className="flex items-center space-x-4 text-sm">
              <div className="flex items-center space-x-2">
                <div className="w-4 h-4 bg-green-500 dark:bg-green-600 rounded"></div>
                <span className="text-gray-600 dark:text-gray-400">Fast & Correct</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-4 h-4 bg-green-200 dark:bg-green-300 rounded"></div>
                <span className="text-gray-600 dark:text-gray-400">Slow & Correct</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-4 h-4 bg-red-600 dark:bg-red-700 rounded"></div>
                <span className="text-gray-600 dark:text-gray-400">Fast & Wrong</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-4 h-4 bg-red-200 dark:bg-red-300 rounded"></div>
                <span className="text-gray-600 dark:text-gray-400">Slow & Wrong</span>
              </div>
            </div>
          </div>
          
          <div className="overflow-x-auto max-h-[80vh]">
            <table className="w-full min-w-[600px]">
              <thead className="sticky top-0 z-20">
                <tr className="bg-gray-50 dark:bg-gray-700">
                  <th className="table-cell font-semibold text-gray-900 dark:text-white sticky left-0 bg-gray-50 dark:bg-gray-700 z-10">
                    Model
                  </th>
                  {prompts.map(prompt => (
                    <th key={prompt} className="table-cell font-semibold text-gray-900 dark:text-white">
                      <div className="max-w-[150px] truncate" title={prompt}>
                        {prompt}
                      </div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {models.map(model => (
                  <tr key={model} className="hover:bg-gray-50 dark:hover:bg-gray-700/30">
                    <td className="table-cell font-medium text-gray-900 dark:text-white sticky left-0 bg-white dark:bg-gray-800 z-10">
                      <div className="max-w-[200px] truncate" title={model}>
                        {model}
                      </div>
                    </td>
                    {prompts.map(prompt => getHeatmapCell(model, prompt))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <Modal
        isOpen={!!selectedCell}
        onClose={() => setSelectedCell(null)}
        title={`${selectedCell?.model} - ${selectedCell?.file}`}
        showNavigation={!!selectedCell}
        onPrev={() => navigateToModel('prev')}
        onNext={() => navigateToModel('next')}
        navLabel={selectedCell ? `${models.indexOf(selectedCell.model) + 1} / ${models.length}` : null}
      >
        {selectedCell && (
          <div className="space-y-4">
            
            <div className="flex items-center space-x-4">
              <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                selectedCell.correct 
                  ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300' 
                  : 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
              }`}>
                {selectedCell.correct ? 'Correct' : 'Incorrect'}
              </span>
              <span className="text-sm text-gray-600 dark:text-gray-400">
                Response Time: {selectedCell.response_time}
              </span>
              {selectedCell.evaluator_model && (
                <span className="text-sm text-gray-600 dark:text-gray-400">
                  Evaluator: {selectedCell.evaluator_model}
                </span>
              )}
            </div>
            
            <div>
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                Prompt:
              </h4>
              <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap max-h-32 overflow-y-auto">
                {selectedCell.prompt}
              </pre>
            </div>

            <div>
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                Expected Answer:
              </h4>
              <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap max-h-32 overflow-y-auto">
                {selectedCell.expected}
              </pre>
            </div>
            
            <div>
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                Generated Answer:
              </h4>
              <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap max-h-96 overflow-y-auto">
                {selectedCell.generated}
              </pre>
            </div>
            
            {selectedCell.note && (
              <div>
                <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                  Evaluation Note:
                </h4>
                <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 p-4 rounded-lg">
                  <p className="text-sm text-yellow-800 dark:text-yellow-300">
                    {selectedCell.note}
                  </p>
                </div>
              </div>
            )}
          </div>
        )}
      </Modal>
    </>
  );
};