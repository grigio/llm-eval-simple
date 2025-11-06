import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'fs'
import path from 'path'

export default defineConfig({
  plugins: [
    react(),
    {
      name: 'configure-server',
      configureServer(server) {
        server.middlewares.use('/api/json', (req, res, next) => {
          if (req.method === 'GET' && req.url) {
            const filename = req.url.substring(1); // Remove leading slash
            const jsonPath = path.resolve(__dirname, '../answers-generated', filename);
            
            try {
              if (fs.existsSync(jsonPath) && filename.endsWith('.json')) {
                const content = fs.readFileSync(jsonPath, 'utf-8');
                res.setHeader('Content-Type', 'application/json');
                res.end(content);
                return;
              }
            } catch (error) {
              console.error('Error serving JSON file:', error);
            }
          }
          next();
        });
      }
    }
  ],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:4000',
        changeOrigin: true
      }
    }
  }
})