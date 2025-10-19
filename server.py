import http.server
import socketserver
import json
import os
from urllib.parse import urlparse, parse_qs

PORT = 8001
EVALUATED_REPORT_PATH = os.path.join("answers-generated", "report-evaluated.json")
TEMPLATE_PATH = "report_template.html"

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
        # Calculate summary
        model_summary = {}
        for r in results:
            model = r["model"]
            if model not in model_summary:
                model_summary[model] = {"total": 0, "correct": 0, "total_time": 0}
            
            model_summary[model]["total"] += 1
            if r["correct"]:
                model_summary[model]["correct"] += 1
            model_summary[model]["total_time"] += r["response_time"]

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

        detailed_results_body = ""
        for model in models:
            detailed_results_body += f"<tr><td>{model}</td>"
            for prompt in prompts:
                cell_style = ""
                response_time_text = ""
                for r in results:
                    if r["model"] == model and r["file"] == prompt:
                        normalized_time = (r["response_time"] - min_time) / time_range
                        response_time_text = f'<div style="text-align: center; font-weight: bold; font-size: 0.9em;">{r["response_time"]:.2f}s</div>'
                        
                        if r["correct"]:
                            r_fast, g_fast, b_fast = 0, 247, 0
                            r_slow, g_slow, b_slow = 245, 255, 245
                            r_val = int(r_fast + (r_slow - r_fast) * normalized_time)
                            g_val = int(g_fast + (g_slow - g_fast) * normalized_time)
                            b_val = int(b_fast + (b_slow - b_fast) * normalized_time)
                            cell_style = f' style="background-color: rgb({r_val}, {g_val}, {b_val});"'
                        else:
                            lightness = int(70 + 30 * normalized_time)
                            cell_style = f' style="background-color: hsl(0, 100%, {lightness}%);"'
                        break
                cell_id = f"{model}-{prompt}"
                detailed_results_body += f'<td{cell_style} data-cell-id="{cell_id}" onclick="showOverlay(\'{cell_id}\')">{response_time_text}</td>'
            detailed_results_body += "</tr>"

        # Questions details
        results_by_file = {}
        for r in results:
            if r['file'] not in results_by_file:
                results_by_file[r['file']] = {
                    "prompt": r['prompt'],
                    "expected": r['expected'],
                    "models": []
                }
            results_by_file[r['file']]['models'].append(r)

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

        # Cell data for JavaScript
        cell_data = ""
        for r in results:
            cell_id = f"{r['model']}-{r['file']}"
            escaped_answer = r['generated'].replace('"', '\\"').replace('\n', '\\n')
            cell_data += f'''
                "{cell_id}": {{
                    "model": "{r["model"]}",
                    "file": "{r["file"]}",
                    "generated": "{escaped_answer}",
                    "response_time": {r["response_time"]:.2f},
                    "correct": {str(r["correct"]).lower()}
                }},'''
        
        return summary_table, detailed_results_header, detailed_results_body, questions_details, f"{{{cell_data.strip(',')}}}"

with socketserver.TCPServer(("", PORT), ReportHandler) as httpd:
    print("serving at port", PORT)
    httpd.serve_forever()
