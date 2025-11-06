import { useState } from 'react';
import { Filter, X } from 'lucide-react';
import { Button } from './ui/Button';
import { useFilters } from '../hooks/useFilters';
import { getUniquePromptsAndModels } from '../utils/dataProcessing';

export const FilterPanel = ({ results, onFiltersChange }) => {
  const { filters, updateFilter, clearFilters } = useFilters();
  const [isOpen, setIsOpen] = useState(false);
  const { prompts, models } = getUniquePromptsAndModels(results);

  const handleFilterChange = (key, value) => {
    updateFilter(key, value);
    onFiltersChange(filters);
  };

  const handleClear = () => {
    clearFilters();
    onFiltersChange({});
  };

  return (
    <div className="card">
      <div className="p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Filter className="w-4 h-4 text-gray-500" />
            <span className="font-medium text-gray-900 dark:text-white">
              Filters
            </span>
            {(filters.model || filters.file || filters.correctness !== undefined) && (
              <span className="bg-primary-100 text-primary-800 dark:bg-primary-900/30 dark:text-primary-300 text-xs px-2 py-1 rounded-full">
                Active
              </span>
            )}
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsOpen(!isOpen)}
          >
            {isOpen ? <X className="w-4 h-4" /> : <Filter className="w-4 h-4" />}
          </Button>
        </div>

        {isOpen && (
          <div className="mt-4 space-y-4 animate-fade-in">
            {/* Model Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Model
              </label>
              <select
                value={filters.model || 'all'}
                onChange={(e) => handleFilterChange('model', e.target.value === 'all' ? null : e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              >
                <option value="all">All Models</option>
                {models.map(model => (
                  <option key={model} value={model}>{model}</option>
                ))}
              </select>
            </div>

            {/* File Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Prompt File
              </label>
              <select
                value={filters.file || 'all'}
                onChange={(e) => handleFilterChange('file', e.target.value === 'all' ? null : e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              >
                <option value="all">All Files</option>
                {prompts.map(prompt => (
                  <option key={prompt} value={prompt}>{prompt}</option>
                ))}
              </select>
            </div>

            {/* Correctness Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Correctness
              </label>
              <select
                value={filters.correctness !== undefined ? filters.correctness : 'all'}
                onChange={(e) => handleFilterChange('correctness', e.target.value === 'all' ? undefined : e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              >
                <option value="all">All Results</option>
                <option value="correct">Correct Only</option>
                <option value="incorrect">Incorrect Only</option>
              </select>
            </div>

            {/* Max Response Time Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Max Response Time (seconds)
              </label>
              <input
                type="number"
                value={filters.maxResponseTime || ''}
                onChange={(e) => handleFilterChange('maxResponseTime', e.target.value ? parseFloat(e.target.value) : null)}
                placeholder="No limit"
                min="0"
                step="0.1"
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              />
            </div>

            {/* Clear Filters Button */}
            <Button
              variant="secondary"
              onClick={handleClear}
              className="w-full"
            >
              Clear All Filters
            </Button>
          </div>
        )}
      </div>
    </div>
  );
};