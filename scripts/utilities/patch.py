import re

def update_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    if 'get_plotly_config' not in code and 'from src.viz.theme' in code:
        code = code.replace(
            'get_plotly_annotation_style',
            'get_plotly_annotation_style, get_plotly_config'
        )

    # Remove the hardcoded config_opts in explainability_hub
    code = re.sub(
        r'\s*config_opts = \{\s*\"displayModeBar\": True,\s*\"toImageButtonOptions\": \{\s*\"format\": \"png\",\s*\"scale\": 3\s*\}\s*\}\s*',
        '\n        ',
        code
    )

    # Replace the bee config
    code = re.sub(
        r'config_opts\[\"toImageButtonOptions\"\]\[\"filename\"\] = f\"shap_beeswarm_\{h3\}\"\s*st\.plotly_chart\(fig_bee, use_container_width=True, config=config_opts\)',
        'st.plotly_chart(fig_bee, use_container_width=True, config=get_plotly_config(f\"shap_beeswarm_{h3}\"))',
        code
    )

    # Replace the dep config
    code = re.sub(
        r'dep_config = config_opts\.copy\(\)\s*dep_config\[\"toImageButtonOptions\"\]\[\"filename\"\] = f\"shap_dep_\{h3\}_\{feature_name\}\"\s*st\.plotly_chart\(fig_dep, use_container_width=True, config=dep_config\)',
        'st.plotly_chart(fig_dep, use_container_width=True, config=get_plotly_config(f\"shap_dep_{h3}_{feature_name}\"))',
        code
    )

    # Fix other st.plotly_chart(fig, use_container_width=True) -> use_container_width=True, config=get_plotly_config()
    code = re.sub(
        r'st\.plotly_chart\(([^,]+),\s*use_container_width=True\)',
        r'st.plotly_chart(\1, use_container_width=True, config=get_plotly_config())',
        code
    )
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(code)

update_file('src/explainability_hub.py')
update_file('src/reporting/charts.py')
try:
    update_file('src/pipeline_walkthrough.py')
except FileNotFoundError:
    pass
