import http.server
import json
import os
from urllib.parse import urlparse, parse_qs

from shared import (
    DEFAULT_SERVER_PORT,
    EVALUATED_REPORT_PATH,
    TEMPLATE_PATH,
    calculate_model_summary,
    create_cell_data_dict,
    find_fastest_correct_per_prompt,
    get_unique_prompts_and_models,
    group_results_by_file
)

SERVER_PORT = DEFAULT_SERVER_PORT

class APIHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        # API endpoint for results
        if parsed_path.path == '/api/results':
            self.handle_api_request()
        else:
            self.send_error(404, "API endpoint not found. Use /api/results for evaluation data.")
    
    def handle_api_request(self):
        """Handle API requests for evaluation results."""
        try:
            with open(EVALUATED_REPORT_PATH, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # Prepare API response with all necessary data
            api_response = {
                'results': results,
                'summary': self.prepare_summary_data(results),
                'matrix': self.prepare_matrix_data(results),
                'details': self.prepare_details_data(results),
                'metadata': {
                    'total_results': len(results),
                    'models': list(set(r['model'] for r in results)),
                    'files': list(set(r['file'] for r in results))
                }
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(api_response).encode('utf-8'))
            
        except FileNotFoundError:
            self.send_error(404, "Evaluation results not found. Please run the evaluation first.")
        except Exception as e:
            self.send_error(500, f"Internal server error: {e}")
    
    def prepare_summary_data(self, results):
        """Prepare summary data for API response."""
        model_summary = calculate_model_summary(results)
        summary_data = []
        
        for model, stats in model_summary.items():
            total = stats["total"]
            correct = stats["correct"]
            total_time = stats["total_time"]
            accuracy = (correct / total) * 100 if total > 0 else 0
            avg_time = total_time / total if total > 0 else 0
            
            summary_data.append({
                'model': model,
                'total': total,
                'correct': correct,
                'accuracy': round(accuracy, 1),
                'avg_response_time': round(avg_time, 2)
            })
        
        return summary_data
    
    def prepare_matrix_data(self, results):
        """Prepare matrix data for API response."""
        prompts, models = get_unique_prompts_and_models(results)
        fastest_correct = find_fastest_correct_per_prompt(results, prompts)
        
        matrix_data = {
            'prompts': prompts,
            'models': models,
            'cells': []
        }
        
        for result in results:
            cell_data = {
                'model': result['model'],
                'file': result['file'],
                'correct': result['correct'],
                'response_time': result['response_time'],
                'is_fastest_correct': (
                    result['correct'] and 
                    result['file'] in fastest_correct and 
                    fastest_correct[result['file']] == result['model']
                )
            }
            matrix_data['cells'].append(cell_data)
        
        return matrix_data
    
    def prepare_details_data(self, results):
        """Prepare detailed question data for API response."""
        return group_results_by_file(results)
    
    

def run_server():
    """Run the API server."""
    import socketserver
    import socket
    
    print(f"🔌 Starting API server at http://localhost:{SERVER_PORT}")
    print(f"📊 API endpoint available at: http://localhost:{SERVER_PORT}/api/results")
    
    # Create server with socket reuse to handle TIME_WAIT state
    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True
    
    with ReusableTCPServer(("", SERVER_PORT), APIHandler) as httpd:
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()