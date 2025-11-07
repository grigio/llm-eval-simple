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
    group_results_by_file,
    get_evaluated_report_path
)
from validation import FileOperationValidator, APIResponseValidator

SERVER_PORT = DEFAULT_SERVER_PORT

class APIHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        try:
            parsed_path = urlparse(self.path)
            
            # Validate path for security
            self._validate_path(parsed_path.path)
            
            # API endpoint for results
            if parsed_path.path == '/api/results':
                self.handle_api_request()
            else:
                self.send_error(404, "API endpoint not found. Use /api/results for evaluation data.")
        except ValueError as e:
            self.send_error(400, f"Invalid request: {e}")
        except Exception as e:
            self.send_error(500, f"Internal server error: {e}")
    
    def _validate_path(self, path: str):
        """Validate request path for security."""
        if not path or not isinstance(path, str):
            raise ValueError("Invalid path")
        
        # Check for path traversal
        if '..' in path or path.startswith('//'):
            raise ValueError("Path traversal detected")
        
        # Only allow specific endpoints
        allowed_endpoints = {'/api/results', '/'}
        if path not in allowed_endpoints and not path.startswith('/static/'):
            raise ValueError("Endpoint not allowed")
    
    def handle_api_request(self):
        """Handle API requests for evaluation results."""
        try:
            # Allow custom report path via query parameter
            parsed_path = urlparse(self.path)
            query_params = parse_qs(parsed_path.query)
            custom_report = query_params.get('report', [None])[0]
            
            # Get the appropriate report path
            report_path_str = get_evaluated_report_path(custom_report)
            
            # Validate file path and size
            report_path = FileOperationValidator.validate_file_path(report_path_str)
            FileOperationValidator.validate_file_size(report_path)
            
            with open(report_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # Validate results structure
            if not isinstance(results, list):
                raise ValueError("Invalid results format")
            
            # Sanitize and validate results
            sanitized_results = []
            for result in results:
                if not isinstance(result, dict):
                    continue
                
                # Validate required fields
                required_fields = {'model', 'file', 'correct', 'response_time'}
                if not required_fields.issubset(result.keys()):
                    continue
                
                # Sanitize content fields
                if 'prompt' in result:
                    result['prompt'] = APIResponseValidator.sanitize_content(result['prompt'], 5000)
                if 'expected' in result:
                    result['expected'] = APIResponseValidator.sanitize_content(result['expected'], 5000)
                if 'generated' in result:
                    result['generated'] = APIResponseValidator.sanitize_content(result['generated'], 5000)
                
                sanitized_results.append(result)
            
            # Prepare API response with all necessary data
            api_response = {
                'results': sanitized_results,
                'summary': self.prepare_summary_data(sanitized_results),
                'matrix': self.prepare_matrix_data(sanitized_results),
                'details': self.prepare_details_data(sanitized_results),
                'metadata': {
                    'total_results': len(sanitized_results),
                    'models': list(set(r['model'] for r in sanitized_results)),
                    'files': list(set(r['file'] for r in sanitized_results))
                }
            }
            
            # Validate response size
            response_json = json.dumps(api_response)
            if len(response_json.encode('utf-8')) > 50 * 1024 * 1024:  # 50MB limit
                raise ValueError("Response too large")
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(response_json.encode('utf-8'))
            
        except FileNotFoundError:
            self.send_error(404, "Evaluation results not found. Please run the evaluation first.")
        except ValueError as e:
            self.send_error(400, f"Validation error: {e}")
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