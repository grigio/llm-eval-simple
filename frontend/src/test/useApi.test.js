import { renderHook, waitFor } from '@testing-library/react'
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { useApi } from '../hooks/useApi.js'

// Mock fetch
global.fetch = vi.fn()

// Mock window.matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
})

describe('useApi Hook', () => {
  beforeEach(() => {
    // Clear fetch mock before each test
    global.fetch.mockClear()
  })

  afterEach(() => {
    // Reset fetch mock after each test
    global.fetch.mockReset()
  })

  it('should initialize with loading state', () => {
    global.fetch.mockResolvedValue({
      ok: true,
      json: async () => ({ data: 'test' })
    })

    const { result } = renderHook(() => useApi('/api/test'))

    expect(result.current.loading).toBe(true)
    expect(result.current.data).toBe(null)
    expect(result.current.error).toBe(null)
  })

  it('should fetch data successfully', async () => {
    const mockData = { results: [], summary: [] }
    global.fetch.mockResolvedValue({
      ok: true,
      json: async () => mockData
    })

    const { result } = renderHook(() => useApi('/api/results'))

    await waitFor(() => {
      expect(result.current.loading).toBe(false)
    })

    expect(result.current.data).toEqual(mockData)
    expect(result.current.error).toBe(null)
    expect(global.fetch).toHaveBeenCalledWith('/api/results')
  })

  it('should handle HTTP errors', async () => {
    global.fetch.mockResolvedValue({
      ok: false,
      status: 404,
      statusText: 'Not Found'
    })

    const { result } = renderHook(() => useApi('/api/results'))

    await waitFor(() => {
      expect(result.current.loading).toBe(false)
    })

    expect(result.current.data).toBe(null)
    expect(result.current.error).toBe('HTTP error! status: 404')
  })

  it('should handle network errors', async () => {
    const networkError = new Error('Network error')
    global.fetch.mockRejectedValue(networkError)

    const { result } = renderHook(() => useApi('/api/results'))

    await waitFor(() => {
      expect(result.current.loading).toBe(false)
    })

    expect(result.current.data).toBe(null)
    expect(result.current.error).toBe(networkError.message)
  })

  it('should not fetch if url is empty', () => {
    renderHook(() => useApi(''))

    expect(global.fetch).not.toHaveBeenCalled()
  })

  it('should refetch data when refetch is called', async () => {
    const mockData = { results: [], summary: [] }
    global.fetch.mockResolvedValue({
      ok: true,
      json: async () => mockData
    })

    const { result } = renderHook(() => useApi('/api/results'))

    await waitFor(() => {
      expect(result.current.loading).toBe(false)
    })

    // Reset fetch mock
    global.fetch.mockClear()

    // Call refetch
    result.current.refetch()

    expect(global.fetch).toHaveBeenCalledWith('/api/results')
  })

  it('should handle JSON parsing errors', async () => {
    global.fetch.mockResolvedValue({
      ok: true,
      json: async () => {
        throw new Error('Invalid JSON')
      }
    })

    const { result } = renderHook(() => useApi('/api/results'))

    await waitFor(() => {
      expect(result.current.loading).toBe(false)
    })

    expect(result.current.data).toBe(null)
    expect(result.current.error).toBe('Invalid JSON')
  })

  it('should handle empty response', async () => {
    global.fetch.mockResolvedValue({
      ok: true,
      json: async () => null
    })

    const { result } = renderHook(() => useApi('/api/results'))

    await waitFor(() => {
      expect(result.current.loading).toBe(false)
    })

    expect(result.current.data).toBe(null)
    expect(result.current.error).toBe(null)
  })

  it('should update data when url changes', async () => {
    const mockData1 = { results: ['data1'] }
    const mockData2 = { results: ['data2'] }

    global.fetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockData1
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockData2
      })

    const { result, rerender } = renderHook(
      ({ url }) => useApi(url),
      { initialProps: { url: '/api/results1' } }
    )

    await waitFor(() => {
      expect(result.current.data).toEqual(mockData1)
    })

    rerender({ url: '/api/results2' })

    await waitFor(() => {
      expect(result.current.data).toEqual(mockData2)
    })

    expect(global.fetch).toHaveBeenCalledTimes(2)
    expect(global.fetch).toHaveBeenNthCalledWith(1, '/api/results1')
    expect(global.fetch).toHaveBeenNthCalledWith(2, '/api/results2')
  })
})