import { useState } from 'react';

export const useFilters = (initialFilters = {}) => {
  const [filters, setFilters] = useState(initialFilters);

  const updateFilter = (key, value) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  const clearFilters = () => {
    setFilters(initialFilters);
  };

  const applyFilters = (data) => {
    if (!data) return [];
    
    return data.filter(item => {
      // Model filter
      if (filters.model && filters.model !== 'all' && item.model !== filters.model) {
        return false;
      }
      
      // File filter
      if (filters.file && filters.file !== 'all' && item.file !== filters.file) {
        return false;
      }
      
      // Correctness filter
      if (filters.correctness !== undefined) {
        if (filters.correctness === 'correct' && !item.correct) return false;
        if (filters.correctness === 'incorrect' && item.correct) return false;
      }
      
      // Response time filter
      if (filters.maxResponseTime && item.response_time > filters.maxResponseTime) {
        return false;
      }
      
      return true;
    });
  };

  return { filters, updateFilter, clearFilters, applyFilters };
};