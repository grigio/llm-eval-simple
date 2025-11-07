import { renderHook, act } from '@testing-library/react'
import { describe, it, expect, beforeEach } from 'vitest'
import { useFilters } from '../hooks/useFilters.js'

describe('useFilters Hook', () => {
  const mockData = {
    results: [
      { model: 'model-a', file: 'test1.txt', correct: true },
      { model: 'model-a', file: 'test2.txt', correct: false },
      { model: 'model-b', file: 'test1.txt', correct: true },
      { model: 'model-b', file: 'test2.txt', correct: true }
    ],
    summary: [
      { model: 'model-a', accuracy: 50 },
      { model: 'model-b', accuracy: 100 }
    ],
    matrix: {
      prompts: ['test1.txt', 'test2.txt'],
      models: ['model-a', 'model-b'],
      cells: []
    },
    details: {},
    metadata: {
      models: ['model-a', 'model-b'],
      files: ['test1.txt', 'test2.txt']
    }
  }

  beforeEach(() => {
    // Reset any state before each test
  })

  it('should initialize with default values', () => {
    const { result } = renderHook(() => useFilters(mockData))

    expect(result.current.selectedModels).toEqual(['model-a', 'model-b'])
    expect(result.current.selectedFiles).toEqual(['test1.txt', 'test2.txt'])
    expect(result.current.showCorrectOnly).toBe(false)
    expect(result.current.showIncorrectOnly).toBe(false)
    expect(result.current.modelOptions).toEqual(['model-a', 'model-b'])
    expect(result.current.fileOptions).toEqual(['test1.txt', 'test2.txt'])
  })

  it('should handle model selection changes', () => {
    const { result } = renderHook(() => useFilters(mockData))

    act(() => {
      result.current.handleModelChange(['model-a'])
    })

    expect(result.current.selectedModels).toEqual(['model-a'])
  })

  it('should handle file selection changes', () => {
    const { result } = renderHook(() => useFilters(mockData))

    act(() => {
      result.current.handleFileChange(['test1.txt'])
    })

    expect(result.current.selectedFiles).toEqual(['test1.txt'])
  })

  it('should handle correct only toggle', () => {
    const { result } = renderHook(() => useFilters(mockData))

    act(() => {
      result.current.handleCorrectOnlyChange(true)
    })

    expect(result.current.showCorrectOnly).toBe(true)
    expect(result.current.showIncorrectOnly).toBe(false)
  })

  it('should handle incorrect only toggle', () => {
    const { result } = renderHook(() => useFilters(mockData))

    act(() => {
      result.current.handleIncorrectOnlyChange(true)
    })

    expect(result.current.showIncorrectOnly).toBe(true)
    expect(result.current.showCorrectOnly).toBe(false)
  })

  it('should clear all filters', () => {
    const { result } = renderHook(() => useFilters(mockData))

    // Set some filters first
    act(() => {
      result.current.handleModelChange(['model-a'])
      result.current.handleFileChange(['test1.txt'])
      result.current.handleCorrectOnlyChange(true)
    })

    // Clear filters
    act(() => {
      result.current.clearFilters()
    })

    expect(result.current.selectedModels).toEqual(['model-a', 'model-b'])
    expect(result.current.selectedFiles).toEqual(['test1.txt', 'test2.txt'])
    expect(result.current.showCorrectOnly).toBe(false)
    expect(result.current.showIncorrectOnly).toBe(false)
  })

  it('should handle empty data gracefully', () => {
    const emptyData = {
      results: [],
      summary: [],
      matrix: { prompts: [], models: [], cells: [] },
      details: {},
      metadata: { models: [], files: [] }
    }

    const { result } = renderHook(() => useFilters(emptyData))

    expect(result.current.selectedModels).toEqual([])
    expect(result.current.selectedFiles).toEqual([])
    expect(result.current.modelOptions).toEqual([])
    expect(result.current.fileOptions).toEqual([])
  })

  it('should not allow both correct and incorrect only filters simultaneously', () => {
    const { result } = renderHook(() => useFilters(mockData))

    // Enable correct only
    act(() => {
      result.current.handleCorrectOnlyChange(true)
    })

    expect(result.current.showCorrectOnly).toBe(true)
    expect(result.current.showIncorrectOnly).toBe(false)

    // Try to enable incorrect only (should disable correct only)
    act(() => {
      result.current.handleIncorrectOnlyChange(true)
    })

    expect(result.current.showCorrectOnly).toBe(false)
    expect(result.current.showIncorrectOnly).toBe(true)
  })

  it('should extract unique models from data', () => {
    const dataWithDuplicates = {
      ...mockData,
      metadata: {
        models: ['model-a', 'model-b', 'model-a'], // duplicate
        files: ['test1.txt', 'test2.txt']
      }
    }

    const { result } = renderHook(() => useFilters(dataWithDuplicates))

    expect(result.current.modelOptions).toEqual(['model-a', 'model-b'])
  })

  it('should extract unique files from data', () => {
    const dataWithDuplicates = {
      ...mockData,
      metadata: {
        models: ['model-a', 'model-b'],
        files: ['test1.txt', 'test2.txt', 'test1.txt'] // duplicate
      }
    }

    const { result } = renderHook(() => useFilters(dataWithDuplicates))

    expect(result.current.fileOptions).toEqual(['test1.txt', 'test2.txt'])
  })

  it('should handle null/undefined data', () => {
    const { result } = renderHook(() => useFilters(null))

    expect(result.current.selectedModels).toEqual([])
    expect(result.current.selectedFiles).toEqual([])
    expect(result.current.modelOptions).toEqual([])
    expect(result.current.fileOptions).toEqual([])
  })
})