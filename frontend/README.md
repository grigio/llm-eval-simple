# LLM Eval Frontend

Modern React dashboard for LLM evaluation results with interactive charts and responsive design.

## Features

- 🎨 **Modern UI** with Tailwind CSS and dark mode support
- 📊 **Interactive Charts** using Chart.js (accuracy, response time, model comparison)
- 📱 **Mobile Responsive** design that works on all devices
- 🔍 **Advanced Filtering** by model, file, correctness, and response time
- ⚡ **Real-time Updates** with API integration
- 🌙 **Dark Mode** toggle with system preference detection
- 📈 **Performance Metrics** with visual progress bars and color coding

## Quick Start

### Prerequisites

- Node.js 16+ and bun
- Python backend server running

### Setup

```bash
# Navigate to frontend directory
cd frontend

# Run the setup script
./setup.sh

# Or manually:
bun install
bun run build
```

### Development

```bash
# Start development server with hot reload
bun run dev

# In another terminal, start the backend
cd ..
python api_server.py

# Visit http://localhost:3000
```

### Production

```bash
# Build for production
bun run build

# Start the enhanced backend server
python api_server.py

# Visit http://localhost:8001
```

## Project Structure

```
frontend/
├── src/
│   ├── components/          # React components
│   │   ├── Charts/        # Chart.js components
│   │   ├── ui/           # Reusable UI components
│   │   ├── Dashboard.jsx  # Main dashboard
│   │   ├── SummaryTable.jsx
│   │   ├── ResultsMatrix.jsx
│   │   └── QuestionDetails.jsx
│   ├── hooks/             # Custom React hooks
│   ├── utils/             # Utility functions
│   └── styles/            # Global styles
├── package.json
├── vite.config.js
└── tailwind.config.js
```

## Components

### Dashboard
Main component that orchestrates all other components and handles data loading.

### Charts
- **AccuracyChart**: Bar chart showing model accuracy percentages
- **ResponseTimeChart**: Bar chart comparing average response times
- **ModelComparisonChart**: Line chart showing performance across all prompts

### Data Display
- **SummaryTable**: Model performance summary with progress bars
- **ResultsMatrix**: Interactive grid with color-coded performance
- **QuestionDetails**: Expandable details for each prompt

### UI Components
- **DarkModeToggle**: Theme switcher with persistence
- **Modal**: Reusable modal component
- **Button**: Styled button with variants
- **FilterPanel**: Advanced filtering options

## Features Explained

### Dark Mode
- Automatic system preference detection
- Manual toggle with localStorage persistence
- Smooth transitions between themes

### Responsive Design
- Mobile-first approach with Tailwind CSS
- Adaptive layouts for different screen sizes
- Touch-friendly interactions

### Interactive Charts
- Hover tooltips with detailed information
- Color-coded data visualization
- Responsive sizing and animations

### Advanced Filtering
- Filter by model, prompt file, correctness
- Response time range filtering
- Real-time filter application

## API Integration

The frontend communicates with the backend via REST API:

- `GET /api/results` - Fetches all evaluation results
- Includes summary statistics, matrix data, and detailed information

## Development Notes

- Uses Vite for fast development and building
- React 18 with modern hooks and patterns
- Chart.js for data visualization
- Tailwind CSS for styling
- Lucide React for icons

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

## Contributing

1. Follow existing code patterns
2. Use TypeScript for new components (optional)
3. Test on mobile devices
4. Ensure dark mode compatibility