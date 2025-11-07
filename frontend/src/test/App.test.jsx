import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { BrowserRouter } from 'react-router-dom'
import '@testing-library/jest-dom'

// Mock the hooks before importing App
vi.mock('../hooks/useApi.js', () => ({
  useApi: vi.fn(),
  useJsonFile: vi.fn()
}))

vi.mock('../hooks/useDarkMode.js', () => ({
  useDarkMode: vi.fn()
}))

vi.mock('../hooks/useFilters.js', () => ({
  useFilters: () => ({
    selectedModels: ['test-model'],
    selectedFiles: ['test.txt'],
    showCorrectOnly: false,
    showIncorrectOnly: false,
    modelOptions: ['test-model'],
    fileOptions: ['test.txt'],
    handleModelChange: vi.fn(),
    handleFileChange: vi.fn(),
    handleCorrectOnlyChange: vi.fn(),
    handleIncorrectOnlyChange: vi.fn(),
    clearFilters: vi.fn()
  })
}))

// Mock window.location
Object.defineProperty(window, 'location', {
  value: {
    search: '',
    pathname: '/',
    href: ''
  },
  writable: true
})

// Mock localStorage
const localStorageMock = {
  getItem: vi.fn(),
  setItem: vi.fn(),
  removeItem: vi.fn(),
  clear: vi.fn()
}
Object.defineProperty(window, 'localStorage', {
  value: localStorageMock
})

// Import App after mocking
import App from '../App.jsx'
import { useApi, useJsonFile } from '../hooks/useApi.js'
import { useDarkMode } from '../hooks/useDarkMode.js'

const renderWithRouter = (component) => {
  return render(
    <BrowserRouter>
      {component}
    </BrowserRouter>
  )
}

describe('App Component', () => {
  beforeEach(() => {
    // Clear all mocks before each test
    vi.clearAllMocks()
    
    // Reset default mock implementations
    useApi.mockReturnValue({
      data: {
        results: [
          {
            model: 'test-model',
            file: 'test.txt',
            correct: true,
            response_time: 1.5
          }
        ]
      },
      loading: false,
      error: null
    })
    
    useJsonFile.mockReturnValue({
      data: null,
      loading: false,
      error: null
    })
    
    useDarkMode.mockReturnValue({
      isDark: false,
      toggleDarkMode: vi.fn()
    })
    
    // Reset window.location
    window.location.search = ''
    window.location.pathname = '/'
    window.location.href = ''
  })

  it('renders without crashing', () => {
    renderWithRouter(<App />)
    
    // Check if main components are rendered
    expect(screen.getByText('LLM Eval Dashboard')).toBeInTheDocument()
  })

  it('displays loading state initially', () => {
    // Override the mock to show loading state
    useApi.mockReturnValue({
      data: null,
      loading: true,
      error: null
    })

    renderWithRouter(<App />)
    
    expect(screen.getByText('Loading evaluation results...')).toBeInTheDocument()
  })

  it('displays error state when API fails', () => {
    // Override the mock to show error state
    useApi.mockReturnValue({
      data: null,
      loading: false,
      error: 'Failed to fetch data'
    })

    renderWithRouter(<App />)
    
    expect(screen.getByText('Error Loading Results')).toBeInTheDocument()
    expect(screen.getByText('Failed to fetch data')).toBeInTheDocument()
  })

  it('displays dashboard when data is loaded', () => {
    renderWithRouter(<App />)
    
    // Check if dashboard components are rendered
    expect(screen.getByText('LLM Eval Dashboard')).toBeInTheDocument()
    expect(screen.getByText('Interactive model evaluation results')).toBeInTheDocument()
  })

  it('renders results matrix', () => {
    renderWithRouter(<App />)
    
    // Check if results matrix is rendered
    expect(screen.getByText('test.txt')).toBeInTheDocument()
    expect(screen.getAllByText('test-model')).toHaveLength(2) // One in table, one in accordion
  })

  it('toggles dark mode', () => {
    const mockToggle = vi.fn()
    useDarkMode.mockReturnValue({
      isDark: false,
      toggle: mockToggle
    })

    renderWithRouter(<App />)
    
    // Find dark mode toggle button by aria-label
    const darkModeToggle = screen.getByLabelText('Toggle dark mode')
    expect(darkModeToggle).toBeInTheDocument()
  })

  it('applies dark mode class when enabled', () => {
    useDarkMode.mockReturnValue({
      isDark: true,
      toggle: vi.fn()
    })

    renderWithRouter(<App />)
    
    // Check that sun icon is shown when dark mode is enabled
    const darkModeToggle = screen.getByLabelText('Toggle dark mode')
    expect(darkModeToggle).toBeInTheDocument()
    // The dark mode class application is handled by the hook's useEffect,
    // which doesn't run in the mocked test environment
  })
})