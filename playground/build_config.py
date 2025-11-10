from nicegui import ui
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict
import yaml
from pathlib import Path
from predefined_options import VULNERABILITY_TYPES, ATTACK_CATEGORIES

# ============================================================================
# Pydantic models for validation
# ============================================================================

class ModelConfig(BaseModel):
    provider: str = "custom"
    file: str = Field(min_length=1, description="Python file path")
    class_: str = Field(alias="class", min_length=1, description="Class name")

class ModelWrapper(BaseModel):
    model: ModelConfig

class ModelsConfig(BaseModel):
    simulator: ModelWrapper
    evaluation: ModelWrapper

class CallbackConfig(BaseModel):
    file: str = Field(min_length=1, description="Python file path")
    function: str = Field(min_length=1, description="Function name")

class TargetConfig(BaseModel):
    purpose: str = Field(min_length=1, description="Target system purpose")
    callback: CallbackConfig

class SystemConfig(BaseModel):
    max_concurrent: int = Field(ge=1, le=100, description="Max concurrent attacks")
    attacks_per_vulnerability_type: int = Field(ge=1, le=20, description="Attacks per vulnerability type")
    run_async: bool = True
    ignore_errors: bool = False
    output_folder: str = Field(min_length=1, description="Output folder path")

class VulnerabilityConfig(BaseModel):
    name: str = Field(min_length=1, description="Vulnerability name")
    types: Optional[List[str]] = None

class AttackConfig(BaseModel):
    name: str = Field(min_length=1, description="Attack name")
    weight: int = Field(ge=1, le=10, description="Attack weight")

class DeepTeamConfig(BaseModel):
    models: ModelsConfig
    target: TargetConfig
    system_config: SystemConfig
    default_vulnerabilities: List[VulnerabilityConfig] = []
    attacks: List[AttackConfig] = []

# ============================================================================
# UI State Management
# ============================================================================

# Track selected vulnerabilities and their sub-types
vulnerability_selections = {}  # {vuln_name: {'enabled': checkbox, 'types': select_widget}}

# Track selected attacks and their weights
attack_selections = {}  # {attack_name: {'enabled': checkbox, 'weight': number_widget}}

def clear_validation_errors():
    """Clear all validation error indicators"""
    sim_model_file.validation_error = None
    sim_model_class.validation_error = None
    eval_model_file.validation_error = None
    eval_model_class.validation_error = None
    
    target_purpose.validation_error = None
    callback_file.validation_error = None
    callback_function.validation_error = None
    
    max_concurrent.validation_error = None
    attacks_per_vuln.validation_error = None
    output_folder.validation_error = None
    
    config_filename.validation_error = None

def submit():
    """Validate and save the configuration"""
    error_area.set_text('')
    success_label.set_text('')
    clear_validation_errors()
    
    # Parse selected vulnerabilities
    vulnerabilities = []
    for vuln_name, widgets in vulnerability_selections.items():
        if widgets['enabled'].value:
            selected_types = widgets['types'].value if widgets['types'] else None
            if selected_types and len(selected_types) > 0:
                vulnerabilities.append({
                    'name': vuln_name,
                    'types': selected_types
                })
            else:
                # No sub-types or none selected
                vulnerabilities.append({
                    'name': vuln_name,
                    'types': None
                })
    
    # Parse selected attacks
    attacks = []
    for attack_name, widgets in attack_selections.items():
        if widgets['enabled'].value:
            attacks.append({
                'name': attack_name,
                'weight': int(widgets['weight'].value)
            })
    
    # Build the configuration payload
    payload = {
        'models': {
            'simulator': {
                'model': {
                    'provider': 'custom',
                    'file': sim_model_file.value.strip(),
                    'class': sim_model_class.value.strip()
                }
            },
            'evaluation': {
                'model': {
                    'provider': 'custom',
                    'file': eval_model_file.value.strip(),
                    'class': eval_model_class.value.strip()
                }
            }
        },
        'target': {
            'purpose': target_purpose.value.strip(),
            'callback': {
                'file': callback_file.value.strip(),
                'function': callback_function.value.strip()
            }
        },
        'system_config': {
            'max_concurrent': int(max_concurrent.value),
            'attacks_per_vulnerability_type': int(attacks_per_vuln.value),
            'run_async': bool(run_async.value),
            'ignore_errors': bool(ignore_errors.value),
            'output_folder': output_folder.value.strip()
        },
        'default_vulnerabilities': vulnerabilities,
        'attacks': attacks
    }
    
    # Validate with Pydantic
    try:
        validated_config = DeepTeamConfig(**payload)
    except ValidationError as e:
        error_lines = []
        for err in e.errors():
            loc = err['loc']
            msg = err['msg']
            error_lines.append(f"• {'.'.join(str(x) for x in loc)}: {msg}")
            
            # Map errors to specific fields
            if loc == ('models', 'simulator', 'model', 'file'):
                sim_model_file.validation_error = msg
            elif loc == ('models', 'simulator', 'model', 'class'):
                sim_model_class.validation_error = msg
            elif loc == ('models', 'evaluation', 'model', 'file'):
                eval_model_file.validation_error = msg
            elif loc == ('models', 'evaluation', 'model', 'class'):
                eval_model_class.validation_error = msg
            elif loc == ('target', 'purpose'):
                target_purpose.validation_error = msg
            elif loc == ('target', 'callback', 'file'):
                callback_file.validation_error = msg
            elif loc == ('target', 'callback', 'function'):
                callback_function.validation_error = msg
            elif loc == ('system_config', 'max_concurrent'):
                max_concurrent.validation_error = msg
            elif loc == ('system_config', 'attacks_per_vulnerability_type'):
                attacks_per_vuln.validation_error = msg
            elif loc == ('system_config', 'output_folder'):
                output_folder.validation_error = msg
        
        error_area.set_text("❌ Validation errors:\n" + "\n".join(error_lines))
        ui.notify('Validation failed. Please fix the errors.', type='negative', position='top')
        return
    
    # Validate that at least one vulnerability or attack is selected
    if len(vulnerabilities) == 0 and len(attacks) == 0:
        error_area.set_text("❌ Please select at least one vulnerability or attack")
        ui.notify('No vulnerabilities or attacks selected', type='warning', position='top')
        return
    
    # Save to file
    filename = config_filename.value.strip()
    if not filename:
        config_filename.validation_error = "Filename is required"
        error_area.set_text("❌ Please provide a filename")
        return
    
    # Ensure .yaml extension
    output_path = Path(filename).with_suffix('.yaml')
    
    try:
        with open(output_path, 'w') as f:
            # Convert to dict and handle the 'class' field alias
            config_dict = validated_config.model_dump(by_alias=True)
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        ui.notify(f'✅ Configuration saved to {output_path}', type='positive', position='top')
        success_label.set_text(f'✅ Successfully saved to: {output_path.absolute()}')
    except Exception as e:
        error_area.set_text(f"❌ Error saving file: {str(e)}")
        ui.notify('Failed to save file', type='negative', position='top')

def load_example():
    """Load example values into the form"""
    sim_model_file.value = 'my_model.py'
    sim_model_class.value = 'CustomDeepEvalLLM'
    eval_model_file.value = 'my_model.py'
    eval_model_class.value = 'CustomDeepEvalLLM'
    
    target_purpose.value = 'A custom chatbot'
    callback_file.value = 'my_callback.py'
    callback_function.value = 'model_callback'
    
    max_concurrent.value = 10
    attacks_per_vuln.value = 3
    run_async.value = True
    ignore_errors.value = False
    output_folder.value = 'results'
    
    # Select example vulnerabilities
    if "Bias" in vulnerability_selections:
        vulnerability_selections["Bias"]['enabled'].value = True
        vulnerability_selections["Bias"]['types'].value = ["age", "race", "gender"]
    
    if "Misinformation" in vulnerability_selections:
        vulnerability_selections["Misinformation"]['enabled'].value = True
        vulnerability_selections["Misinformation"]['types'].value = ["financial"]
    
    if "PII" in vulnerability_selections:
        vulnerability_selections["PII"]['enabled'].value = True
        vulnerability_selections["PII"]['types'].value = ["social_security", "credit_card"]
    
    if "Excessive Agency" in vulnerability_selections:
        vulnerability_selections["Excessive Agency"]['enabled'].value = True
    
    # Select example attacks
    if "Prompt Injection" in attack_selections:
        attack_selections["Prompt Injection"]['enabled'].value = True
        attack_selections["Prompt Injection"]['weight'].value = 4
    
    if "Jailbreaking" in attack_selections:
        attack_selections["Jailbreaking"]['enabled'].value = True
        attack_selections["Jailbreaking"]['weight'].value = 3
    
    if "Context Poisoning" in attack_selections:
        attack_selections["Context Poisoning"]['enabled'].value = True
        attack_selections["Context Poisoning"]['weight'].value = 2
    
    if "ROT13" in attack_selections:
        attack_selections["ROT13"]['enabled'].value = True
        attack_selections["ROT13"]['weight'].value = 1
    
    config_filename.value = 'deepteam_config.yaml'
    
    ui.notify('Example configuration loaded', type='info', position='top')


# ============================================================================
# Build the UI
# ============================================================================

ui.colors(primary='#2563eb', secondary='#7c3aed', accent='#db2777', positive='#16a34a', negative='#dc2626')

with ui.column().classes('w-full max-w-5xl mx-auto p-4'):
    ui.label('DeepTeam Configuration Builder').classes('text-3xl font-bold mb-4')
    ui.label('Build YAML configuration for red teaming models').classes('text-gray-600 mb-6')
    
    with ui.row().classes('w-full gap-2 mb-6'):
        ui.button('Load Example', on_click=load_example, color='secondary', icon='file_download')
    
    # Models Section
    with ui.expansion('🤖 Models Configuration', icon='smart_toy').classes('w-full mb-4').props('default-opened'):
        with ui.card().classes('w-full'):
            ui.label('Simulator Model (Payload Generation)').classes('text-lg font-semibold mb-2')
            with ui.row().classes('w-full gap-4'):
                sim_model_file = ui.input('Model File', placeholder='e.g., my_model.py') \
                    .props('dense outlined').classes('flex-grow')
                sim_model_class = ui.input('Model Class', placeholder='e.g., CustomDeepEvalLLM') \
                    .props('dense outlined').classes('flex-grow')
            
            ui.separator()
            
            ui.label('Evaluation Model (Output Evaluation)').classes('text-lg font-semibold mb-2 mt-4')
            with ui.row().classes('w-full gap-4'):
                eval_model_file = ui.input('Model File', placeholder='e.g., my_model.py') \
                    .props('dense outlined').classes('flex-grow')
                eval_model_class = ui.input('Model Class', placeholder='e.g., CustomDeepEvalLLM') \
                    .props('dense outlined').classes('flex-grow')
    
    # Target Section
    with ui.expansion('🎯 Target System Configuration', icon='target').classes('w-full mb-4').props('default-opened'):
        with ui.card().classes('w-full'):
            target_purpose = ui.input('Target Purpose', placeholder='e.g., A custom chatbot') \
                .props('dense outlined').classes('w-full mb-4')
            
            ui.label('Callback Configuration').classes('text-lg font-semibold mb-2')
            with ui.row().classes('w-full gap-4'):
                callback_file = ui.input('Callback File', placeholder='e.g., my_callback.py') \
                    .props('dense outlined').classes('flex-grow')
                callback_function = ui.input('Callback Function', placeholder='e.g., model_callback') \
                    .props('dense outlined').classes('flex-grow')
    
    # System Configuration
    with ui.expansion('⚙️ System Configuration', icon='settings').classes('w-full mb-4').props('default-opened'):
        with ui.card().classes('w-full'):
            with ui.row().classes('w-full gap-4'):
                max_concurrent = ui.number('Max Concurrent', value=10, min=1, max=100, precision=0) \
                    .props('dense outlined').classes('flex-grow')
                attacks_per_vuln = ui.number('Attacks Per Vulnerability', value=3, min=1, max=20, precision=0) \
                    .props('dense outlined').classes('flex-grow')
            
            output_folder = ui.input('Output Folder', value='results', placeholder='results') \
                .props('dense outlined').classes('w-full mt-4')
            
            with ui.row().classes('w-full gap-4 mt-4'):
                run_async = ui.checkbox('Run Async', value=True)
                ignore_errors = ui.checkbox('Ignore Errors', value=False)
    
    # Vulnerabilities Section
    with ui.expansion('🛡️ Default Vulnerabilities', icon='security').classes('w-full mb-4').props('default-opened'):
        with ui.card().classes('w-full'):
            ui.label('Select vulnerability types to test').classes('text-sm text-gray-600 mb-3')
            
            with ui.column().classes('w-full gap-3'):
                for vuln_name, sub_types in VULNERABILITY_TYPES.items():
                    with ui.card().classes('w-full').props('flat bordered'):
                        with ui.row().classes('w-full items-start gap-4'):
                            # Checkbox to enable/disable this vulnerability
                            enabled_cb = ui.checkbox(vuln_name, value=False).classes('text-lg font-medium')
                            
                            # If there are sub-types, show a multi-select
                            if sub_types:
                                types_select = ui.select(
                                    sub_types,
                                    label=f'{vuln_name} Sub-types',
                                    multiple=True,
                                    value=[]
                                ).props('dense outlined use-chips').classes('flex-grow')
                                
                                vulnerability_selections[vuln_name] = {
                                    'enabled': enabled_cb,
                                    'types': types_select
                                }
                            else:
                                ui.label('(No sub-types)').classes('text-gray-400 text-sm self-center')
                                vulnerability_selections[vuln_name] = {
                                    'enabled': enabled_cb,
                                    'types': None
                                }
    
    # Attacks Section
    with ui.expansion('⚔️ Attacks Configuration', icon='gavel').classes('w-full mb-4').props('default-opened'):
        with ui.card().classes('w-full'):
            ui.label('Select attack types and assign weights (1-10)').classes('text-sm text-gray-600 mb-3')
            
            with ui.column().classes('w-full gap-4'):
                for category_name, attack_methods in ATTACK_CATEGORIES.items():
                    ui.label(f'📌 {category_name}').classes('text-lg font-semibold mt-2')
                    
                    with ui.column().classes('w-full gap-2 ml-4'):
                        for attack_name in attack_methods:
                            with ui.card().classes('w-full').props('flat bordered'):
                                with ui.row().classes('w-full items-center gap-4'):
                                    enabled_cb = ui.checkbox(attack_name, value=False).classes('flex-grow')
                                    weight_input = ui.number(
                                        'Weight',
                                        value=1,
                                        min=1,
                                        max=10,
                                        precision=0
                                    ).props('dense outlined').classes('w-32')
                                    
                                    attack_selections[attack_name] = {
                                        'enabled': enabled_cb,
                                        'weight': weight_input
                                    }
    
    # Output Configuration
    ui.separator()
    
    with ui.card().classes('w-full mt-6'):
        ui.label('📄 Output Configuration').classes('text-lg font-semibold mb-2')
        config_filename = ui.input('Config Filename', value='deepteam_config.yaml', 
                                   placeholder='deepteam_config.yaml') \
            .props('dense outlined').classes('w-full')
    
    # Error and Success Messages
    error_area = ui.label('').style('white-space: pre-wrap; color: #dc2626; font-family: monospace').classes('mt-4')
    success_label = ui.label('').style('color: #16a34a; font-weight: bold').classes('mt-2')
    
    # Submit Button
    with ui.row().classes('w-full mt-6 gap-2'):
        ui.button('💾 Save Configuration', on_click=submit, color='positive', icon='save').classes('text-lg px-6 py-3')

ui.run(title='DeepTeam Config Builder', reload=False, port=8080)
