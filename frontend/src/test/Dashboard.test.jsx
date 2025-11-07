import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { BrowserRouter } from 'react-router-dom'
import { Dashboard } from '../components/Dashboard.jsx'
import '@testing-library/jest-dom'

// Mock localStorage
const localStorageMock = {
  getItem: vi.fn(),
  setItem: vi.fn(),
  removeItem: vi.fn(),
  clear: vi.fn(),
};
global.localStorage = localStorageMock

// Mock the hooks
vi.mock('../hooks/useApi.js', () => ({
  useApi: () => ({
    data: {
      results: [
        {
          model: 'test-model',
          file: 'test.txt',
          correct: true,
          response_time: 1.5
        }
      ],
      summary: [
        {
          model: 'test-model',
          total: 1,
          correct: 1,
          accuracy: 100.0,
          avg_response_time: 1.5
        }
      ],
      matrix: {
        prompts: ['test.txt'],
        models: ['test-model'],
        cells: [
          {
            model: 'test-model',
            file: 'test.txt',
            correct: true,
            response_time: 1.5,
            is_fastest_correct: true
          }
        ]
      },
      details: {
        'test.txt': {
          prompt: 'Test prompt',
          expected: 'Expected answer',
          models: [
            {
              model: 'test-model',
              file: 'test.txt',
              correct: true,
              response_time: 1.5,
              generated: 'Generated answer'
            }
          ]
        }
      },
      metadata: {
        total_results: 1,
        models: ['test-model'],
        files: ['test.txt']
      }
    },
    loading: false,
    error: null
  }),
  useJsonFile: () => ({
    data: null,
    loading: false,
    error: null
  })
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

const renderWithRouter = (component) => {
  return render(
    <BrowserRouter>
      {component}
    </BrowserRouter>
  )
}

describe('Dashboard Component', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders dashboard title', () => {
    renderWithRouter(<Dashboard />)
    
    expect(screen.getByText('LLM Eval Dashboard')).toBeInTheDocument()
  })

  it('renders evaluation heatmap', () => {
    renderWithRouter(<Dashboard />)
    
    expect(screen.getByText('Evaluation Heatmap')).toBeInTheDocument()
    expect(screen.getByText('test.txt')).toBeInTheDocument()
    expect(screen.getAllByText('test-model')).toHaveLength(2) // Appears in both table and accordion
  })

  it('renders model answers details', () => {
    renderWithRouter(<Dashboard />)
    
    expect(screen.getByText('Model Answers Details')).toBeInTheDocument()
  })

  it('renders footer', () => {
    renderWithRouter(<Dashboard />)
    
    expect(screen.getByText('LLM Eval Simple - Interactive Dashboard')).toBeInTheDocument()
  })

  // Note: Tests for loading, error, and no-data states require dynamic mocking
  // which is complex with vitest. These states are tested indirectly through the App tests.
})