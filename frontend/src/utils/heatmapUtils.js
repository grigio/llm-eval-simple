export const getHeatmapColor = (correct, responseTime, minTime, maxTime, isDarkMode) => {
  // Normalize response time between 0 and 1
  const normalizedTime = maxTime > minTime 
    ? (responseTime - minTime) / (maxTime - minTime)
    : 0.5;

  // Color intensity based on correctness and speed
  // Fast results = darker colors, Slow results = lighter colors
  // Correct + Fast = Dark Green
  // Correct + Slow = Light Green  
  // Incorrect + Fast = Dark Red
  // Incorrect + Slow = Light Red
  
  if (correct) {
    // Green colors
    if (normalizedTime < 0.5) {
      // Fast and correct - dark green
      return {
        bg: isDarkMode ? 'rgb(22, 163, 74)' : 'rgb(34, 197, 94)', // green-600/500
        hover: isDarkMode ? 'rgb(21, 128, 61)' : 'rgb(22, 163, 74)', // green-700/600 (darker on hover)
        text: 'white',
        intensity: 'high'
      };
    } else {
      // Slow and correct - light green
      return {
        bg: isDarkMode ? 'rgb(134, 239, 172)' : 'rgb(187, 247, 208)', // green-300/200
        hover: isDarkMode ? 'rgb(74, 222, 128)' : 'rgb(134, 239, 172)', // green-400/300 (darker on hover)
        text: isDarkMode ? 'rgb(21, 128, 61)' : 'rgb(21, 128, 61)', // green-800
        intensity: 'low'
      };
    }
  } else {
    // Red colors
    if (normalizedTime < 0.5) {
      // Fast but wrong - dark red
      return {
        bg: isDarkMode ? 'rgb(185, 28, 28)' : 'rgb(220, 38, 38)', // red-700/600
        hover: isDarkMode ? 'rgb(153, 27, 27)' : 'rgb(185, 28, 28)', // red-800/700 (darker on hover)
        text: 'white',
        intensity: 'high'
      };
    } else {
      // Slow and wrong - light red
      return {
        bg: isDarkMode ? 'rgb(252, 165, 165)' : 'rgb(254, 202, 202)', // red-300/200
        hover: isDarkMode ? 'rgb(248, 113, 113)' : 'rgb(252, 165, 165)', // red-400/300 (darker on hover)
        text: isDarkMode ? 'rgb(127, 29, 29)' : 'rgb(127, 29, 29)', // red-800
        intensity: 'low'
      };
    }
  }
};

export const getResponseTimeStats = (results) => {
  if (!results || results.length === 0) {
    return { min: 0, max: 1, avg: 0 };
  }

  const times = results
    .map(r => r.response_time)
    .filter(time => typeof time === 'number' && time > 0);

  if (times.length === 0) {
    return { min: 0, max: 1, avg: 0 };
  }

  const min = Math.min(...times);
  const max = Math.max(...times);
  const avg = times.reduce((sum, time) => sum + time, 0) / times.length;

  return { min, max, avg };
};