import { useState, useMemo, useEffect } from 'react';

export const useFilters = (data) => {
  const [selectedModels, setSelectedModels] = useState([]);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [showCorrectOnly, setShowCorrectOnly] = useState(false);
  const [showIncorrectOnly, setShowIncorrectOnly] = useState(false);

  // Extract unique models from data
  const modelOptions = useMemo(() => {
    if (!data || !data.results) return [];
    const models = [...new Set(data.results.map(item => item.model))];
    return models;
  }, [data]);

  // Extract unique files from data
  const fileOptions = useMemo(() => {
    if (!data || !data.results) return [];
    const files = [...new Set(data.results.map(item => item.file))];
    return files;
  }, [data]);

  // Initialize selections on first load
  const [initialized, setInitialized] = useState(false);
  
  useEffect(() => {
    if (data && data.results && !initialized) {
      setSelectedModels(modelOptions);
      setSelectedFiles(fileOptions);
      setInitialized(true);
    }
  }, [data, modelOptions, fileOptions, initialized]);

  const handleModelChange = (models) => {
    setSelectedModels(models);
  };

  const handleFileChange = (files) => {
    setSelectedFiles(files);
  };

  const handleCorrectOnlyChange = (show) => {
    setShowCorrectOnly(show);
    if (show) {
      setShowIncorrectOnly(false);
    }
  };

  const handleIncorrectOnlyChange = (show) => {
    setShowIncorrectOnly(show);
    if (show) {
      setShowCorrectOnly(false);
    }
  };

  const clearFilters = () => {
    setSelectedModels(modelOptions);
    setSelectedFiles(fileOptions);
    setShowCorrectOnly(false);
    setShowIncorrectOnly(false);
  };

  // Apply filters to data
  const filteredData = useMemo(() => {
    if (!data || !data.results) return [];
    
    return data.results.filter(item => {
      // Model filter
      if (selectedModels.length > 0 && !selectedModels.includes(item.model)) {
        return false;
      }
      
      // File filter
      if (selectedFiles.length > 0 && !selectedFiles.includes(item.file)) {
        return false;
      }
      
      // Correctness filter
      if (showCorrectOnly && !item.correct) return false;
      if (showIncorrectOnly && item.correct) return false;
      
      return true;
    });
  }, [data, selectedModels, selectedFiles, showCorrectOnly, showIncorrectOnly]);

  return {
    selectedModels,
    selectedFiles,
    showCorrectOnly,
    showIncorrectOnly,
    modelOptions,
    fileOptions,
    handleModelChange,
    handleFileChange,
    handleCorrectOnlyChange,
    handleIncorrectOnlyChange,
    clearFilters,
    filteredData
  };
};