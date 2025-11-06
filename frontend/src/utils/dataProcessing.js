// Utility functions for data processing (ported from Python)

export const calculateModelSummary = (results) => {
  const modelSummary = {};
  for (const r of results) {
    const model = r.model;
    if (!modelSummary[model]) {
      modelSummary[model] = { total: 0, correct: 0, totalTime: 0 };
    }
    
    modelSummary[model].total += 1;
    if (r.correct) {
      modelSummary[model].correct += 1;
    }
    modelSummary[model].totalTime += r.response_time;
  }
  
  return modelSummary;
};

export const getUniquePromptsAndModels = (results) => {
  const prompts = [...new Set(results.map(r => r.file))].sort();
  const models = [...new Set(results.map(r => r.model))].sort();
  return { prompts, models };
};

export const formatResponseTime = (responseTime) => {
  return `${responseTime.toFixed(2)}s`;
};

export const formatAccuracy = (correct, total) => {
  const accuracy = total > 0 ? (correct / total) * 100 : 0;
  return `${accuracy.toFixed(1)}%`;
};

export const createProgressBar = (accuracy, length = 10) => {
  const barLength = Math.floor(accuracy / 10);
  return '█'.repeat(barLength) + '░'.repeat(length - barLength);
};

export const normalizeTimeValue = (value, minVal, maxVal) => {
  const timeRange = maxVal !== minVal ? maxVal - minVal : 1;
  return (value - minVal) / timeRange;
};

export const interpolateColor = (color1, color2, factor) => {
  const r = Math.round(color1[0] + (color2[0] - color1[0]) * factor);
  const g = Math.round(color1[1] + (color2[1] - color1[1]) * factor);
  const b = Math.round(color1[2] + (color2[2] - color1[2]) * factor);
  return `rgb(${r}, ${g}, ${b})`;
};

export const findFastestCorrectPerPrompt = (results, prompts) => {
  const fastestCorrectPerPrompt = {};
  for (const prompt of prompts) {
    let fastestTime = Infinity;
    let fastestModel = null;
    for (const r of results) {
      if (r.file === prompt && r.correct && r.response_time < fastestTime) {
        fastestTime = r.response_time;
        fastestModel = r.model;
      }
    }
    if (fastestModel) {
      fastestCorrectPerPrompt[prompt] = fastestModel;
    }
  }
  return fastestCorrectPerPrompt;
};

export const groupResultsByFile = (results) => {
  const resultsByFile = {};
  for (const r of results) {
    if (!resultsByFile[r.file]) {
      resultsByFile[r.file] = {
        prompt: r.prompt,
        expected: r.expected,
        models: []
      };
    }
    resultsByFile[r.file].models.push(r);
  }
  return resultsByFile;
};

export const createCellDataDict = (results) => {
  const cellDataDict = {};
  for (const r of results) {
    const cellId = `${r.model}-${r.file}`;
    cellDataDict[cellId] = {
      model: r.model,
      file: r.file,
      generated: r.generated,
      response_time: formatResponseTime(r.response_time),
      correct: r.correct
    };
  }
  return cellDataDict;
};

// Color constants (ported from Python)
export const GOLD_RGB = [255, 215, 0];
export const GREEN_RGB = [0, 247, 0];
export const LIGHT_GREEN_RGB = [245, 255, 245];

export const getCellStyle = (result, fastestCorrectPerPrompt, minTime, maxTime, isDarkMode) => {
  try {
    if (!result) return '';
    
    if (typeof result.response_time !== 'number' || isNaN(result.response_time)) {
      return 'background-color: #fef3c7;'; // Yellow for invalid data
    }
    
    const normalizedTime = normalizeTimeValue(result.response_time, minTime, maxTime);
    const isFastestCorrect = result.correct && 
      fastestCorrectPerPrompt && 
      fastestCorrectPerPrompt[result.file] === result.model;
    
    if (result.correct) {
      if (isFastestCorrect) {
        const color = interpolateColor(GOLD_RGB, GREEN_RGB, normalizedTime);
        return `background-color: ${color}; border: 2px solid #FFD700; box-shadow: 0 0 5px rgba(255, 215, 0, 0.5);`;
      } else {
        const color = interpolateColor(GREEN_RGB, LIGHT_GREEN_RGB, normalizedTime);
        return `background-color: ${color};`;
      }
    } else {
      const lightness = isDarkMode ? 30 : 70;
      const adjustedLightness = lightness + (30 * normalizedTime);
      return `background-color: hsl(0, 100%, ${adjustedLightness}%);`;
    }
  } catch (error) {
    console.error('Error in getCellStyle:', error, { result, fastestCorrectPerPrompt, minTime, maxTime, isDarkMode });
    return 'background-color: #fecaca;'; // Red for errors
  }
};