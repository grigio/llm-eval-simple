import http.server
import json
import os
import socketserver
from urllib.parse import parse_qs, urlparse

from shared import (
    DEFAULT_SERVER_PORT,
    EVALUATED_REPORT_PATH,
    TEMPLATE_PATH,
    GOLD_RGB,
    GREEN_RGB,
    HSL_LIGHTNESS_MIN,
    HSL_LIGHTNESS_RANGE,
    LIGHT_GREEN_RGB,
    RGB_MAX,
    calculate_model_summary,
    create_cell_data_dict,
    find_fastest_correct_per_prompt,
    format_accuracy,
    format_response_time,
    get_unique_prompts_and_models,
    group_results_by_file,
    interpolate_color,
    normalize_time_value
)

SERVER_PORT = DEFAULT_SERVER_PORT

class ReportHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/?render=report-evaluated.json')
            self.end_headers()
            return

        parsed_path = urlparse(self.path)
        if parsed_path.path == '/':
            query_components = parse_qs(parsed_path.query)
            json_file = query_components.get("render", [None])[0]
            if json_file:
                json_path = os.path.join("answers-generated", json_file)
            else:
                json_path = EVALUATED_REPORT_PATH
            
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
                    template = f.read()

                summary_table, detailed_results_header, detailed_results_body, questions_details, cell_data = self.format_results(results)

                html = template.replace("__SUMMARY_TABLE__", summary_table)
                html = html.replace("__DETAILED_RESULTS_HEADER__", detailed_results_header)
                html = html.replace("__DETAILED_RESULTS_BODY__", detailed_results_body)
                html = html.replace("__QUESTIONS_DETAILS__", questions_details)
                html = html.replace("__CELL_DATA__", cell_data)

                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(html.encode('utf-8'))
            except FileNotFoundError:
                self.send_error(404, "Report file not found. Please run the evaluation first.")
            except Exception as e:
                self.send_error(500, f"An error occurred: {e}")
        else:
            super().do_GET()

    def format_results(self, results):
        # Calculate summary using shared function
        model_summary = calculate_model_summary(results)

        summary_table = ""
        for model, stats in model_summary.items():
            total = stats["total"]
            correct = stats["correct"]
            total_time = stats["total_time"]
            accuracy = (correct / total) * 100 if total > 0 else 0
            avg_time = total_time / total if total > 0 else 0
            summary_table += f"""
                <tr>
                    <td>{model}</td>
                    <td>
                        {correct}/{total} ({accuracy:.1f}%)
                        <div class="summary-bar">
                            <div class="summary-bar-fill" style="width: {accuracy:.1f}%;"></div>
                        </div>
                    </td>
                    <td>{avg_time:.2f}s</td>
                </tr>
    """

        # Detailed results grid
        prompts = sorted(list(set(r["file"] for r in results)))
        models = sorted(list(set(r["model"] for r in results)))

        all_response_times = [r["response_time"] for r in results]
        min_time = min(all_response_times)
        max_time = max(all_response_times)
        time_range = max_time - min_time if max_time != min_time else 1

        detailed_results_header = ""
        for prompt in prompts:
            detailed_results_header += f"<th>{prompt}</th>"

        # Find fastest correct test for each prompt file using shared function
        fastest_correct_per_prompt = find_fastest_correct_per_prompt(results, prompts)

        detailed_results_body = ""
        for model in models:
            detailed_results_body += f"<tr><td>{model}</td>"
            for prompt in prompts:
                cell_style = ""
                response_time_text = ""
                is_fastest_correct = False
                
                for r in results:
                    if r["model"] == model and r["file"] == prompt:
                        normalized_time = (r["response_time"] - min_time) / time_range
                        response_time_text = f'<div style="text-align: center; font-weight: bold; font-size: 0.9em;">{r["response_time"]:.2f}s</div>'
                        
                        # Check if this is the fastest correct test for this prompt
                        is_fastest_correct = (r["correct"] and 
                                            prompt in fastest_correct_per_prompt and 
                                            fastest_correct_per_prompt[prompt] == model)
                        
                        if r["correct"]:
                            if is_fastest_correct:
                                # Highlight fastest correct with gold/yellow
                                r_val, g_val, b_val = interpolate_color(GOLD_RGB, GREEN_RGB, normalized_time)
                                cell_style = f' style="background-color: rgb({r_val}, {g_val}, {b_val}); border: 2px solid #FFD700; box-shadow: 0 0 5px rgba(255, 215, 0, 0.5);"'
                            else:
                                # Regular correct answers
                                r_val, g_val, b_val = interpolate_color(GREEN_RGB, LIGHT_GREEN_RGB, normalized_time)
                                cell_style = f' style="background-color: rgb({r_val}, {g_val}, {b_val});"'
                        else:
                            lightness = int(HSL_LIGHTNESS_MIN + HSL_LIGHTNESS_RANGE * normalized_time)
                            cell_style = f' style="background-color: hsl(0, 100%, {lightness}%);"'
                        break
                        
                if is_fastest_correct:
                    response_time_text = '<div style="text-align: center; font-weight: bold; font-size: 0.9em;">⭐ ' + response_time_text.split('>')[1] if '>' in response_time_text else response_time_text
                    
                cell_id = f"{model}-{prompt}"
                detailed_results_body += f'<td{cell_style} data-cell-id="{cell_id}" onclick="showOverlay(\'{cell_id}\')">{response_time_text}</td>'
            detailed_results_body += "</tr>"

        # Questions details using shared function
        results_by_file = group_results_by_file(results)

        questions_details = ""
        for file, data in results_by_file.items():
            questions_details += f"""
            <div class="question">
                <div class="question-header" onclick="toggleDetails(this)">&#9654; {file}</div>
                <div class="models-container">
                    <p><strong>Prompt:</strong></p>
                    <pre>{data['prompt']}</pre>
                    <p><strong>Expected Answer:</strong></p>
                    <pre>{data['expected']}</pre>
                    <hr>
    """
            for model_result in data['models']:
                correct_class = "correct" if model_result['correct'] else "incorrect"
                questions_details += f"""
                    <div class="model-answer {correct_class}">
                        <h4>{model_result['model']}</h4>
                        <p><strong>Generated Answer:</strong></p>
                        <pre>{model_result['generated']}</pre>
                        <p><em>Response Time: {model_result['response_time']:.2f}s</em></p>
                    </div>
    """
            questions_details += """
                </div>
            </div>
    """

        # Cell data for JavaScript using shared function
        cell_data_dict = create_cell_data_dict(results)
        cell_data = json.dumps(json.dumps(cell_data_dict))

        return summary_table, detailed_results_header, detailed_results_body, questions_details, cell_data

if __name__ == "__main__":
    with socketserver.TCPServer(("", SERVER_PORT), ReportHandler) as httpd:
        print("serving at port", SERVER_PORT)
        httpd.serve_forever()
