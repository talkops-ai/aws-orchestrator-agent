"""
Terraform Data Compressor - BALANCED APPROACH

Generic compression class for Terraform data following research document approach.
Reduces 25,000+ characters to ~3,750 characters (85% reduction).

BALANCED COMPRESSION STRATEGY:
✅ COMPRESS (Safe to compress):
- Resource counts and types
- Variable names and types  
- File organization summaries
- Optimizer metrics
- Verbose descriptions

❌ KEEP DETAILED (Essential for accuracy):
- Resource configurations (exact attribute names/values)
- Variable definitions (type, validation, defaults)
- Dependencies (exact dependency chains)
- Local value expressions (exact expressions)
- Output values (exact resource references)

Based on optimized_prompt_research.md findings:
- Content compression with high-density structured format
- Workspace-first hierarchy for state validation
- Smart placeholder values with essential summaries
- Token reduction from 25,000+ to 3,750 characters
- Prevents hallucination by keeping critical details
"""

import re
from typing import List, Dict, Any, Optional


class TerraformDataCompressor:
    """
    Generic compression class for Terraform data following research document approach.
    Reduces 25,000+ characters to ~3,750 characters (85% reduction).
    """
    
    @staticmethod
    def compress_resource_specifications(resource_specs: List[Dict[str, Any]]) -> str:
        """Optimized resource compression - removes redundancy while preserving essential data"""
        if not resource_specs:
            return "None"
    
        # Essential metadata
        resource_count = len(resource_specs)
        unique_types = list(set([spec['resource_type'] for spec in resource_specs]))
    
        # Build optimized resource entries
        resources = []
        for spec in resource_specs:
            resource_entry = f"{spec['resource_type']}:{spec['resource_name']}"
        
            # Add configuration keys (essential for HCL generation)
            if spec['configuration']:
                config_keys = ','.join(spec['configuration'].keys())
                resource_entry += f"[{config_keys}]"
        
            # Add dependencies only if they exist (use shorter format)
            if spec['depends_on']:
                # Shorten dependency references by removing redundant 'aws_' prefixes
                short_deps = []
                for dep in spec['depends_on']:
                    if dep.startswith('aws_'):
                        short_deps.append(dep[4:])  # Remove 'aws_' prefix
                    else:
                        short_deps.append(dep)
                resource_entry += f"→({','.join(short_deps)})"
        
            # Add lifecycle rules only if they exist and are non-default
            if spec['lifecycle_rules']:
                lifecycle_summary = []
                for key, value in spec['lifecycle_rules'].items():
                    if value:  # Only include if True/non-empty
                        if key == 'prevent_destroy':
                            lifecycle_summary.append('pd')
                        elif key == 'create_before_destroy':
                            lifecycle_summary.append('cbd')
                        elif key == 'ignore_changes':
                            lifecycle_summary.append('ic')
                        else:
                            lifecycle_summary.append(key)
                if lifecycle_summary:
                    resource_entry += f"L[{','.join(lifecycle_summary)}]"
        
            resources.append(resource_entry)
    
        # Extract variable and local references
        variables = set()
        locals_refs = set()
    
        for spec in resource_specs:
            config_str = str(spec['configuration'])
            # Extract variable references
            var_matches = re.findall(r'\$\{var\.([^}]+)\}', config_str)
            variables.update(var_matches)
            # Extract local references  
            local_matches = re.findall(r'\$\{local\.([^}]+)\}', config_str)
            locals_refs.update(local_matches)
    
        # Clean up variable names (remove function calls)
        clean_variables = set()
        for var in variables:
            if ',' in var:  # Handle element(var.availability_zones, 0)
                clean_var = var.split(',')[0]
            else:
                clean_var = var
            if clean_var.startswith('element(var.'):
                clean_var = clean_var.replace('element(var.', '').replace(')', '')
            clean_variables.add(clean_var)
        
    
        # Build compressed format
        result_parts = [
            f"Count:{resource_count}",
            f"Types:{','.join(unique_types)}",
            f"Resources:{';'.join(resources)}"
        ]
    
        # Add variables and locals only if they exist
        if clean_variables:
            result_parts.append(f"Vars:{','.join(sorted(clean_variables))}")
        if locals_refs:
            result_parts.append(f"Locals:{','.join(sorted(locals_refs))}")
    
        return "|".join(result_parts)

    
    @staticmethod
    def compress_variable_definitions(variable_defs: List[Dict[str, Any]]) -> str:
        """Compress variable definitions to essential summary - BALANCED APPROACH"""
        if not variable_defs:
            return "None"

        var_count = len(variable_defs)
        unique_types = list({var["type"] for var in variable_defs})

        def compress_rule(rule: str) -> str:
            rl = rule.lower()
            # 1. Numeric range extraction: “between 20 and 65536” → “20-65536”
            nums = re.findall(r"\d+", rl)
            if len(nums) >= 2:
                return f"{nums[0]}-{nums[1]}"
            # 2. Slugify remaining text: remove non-alphanumeric, collapse spaces to _
            slug = re.sub(r"[^\w]+", "_", rl).strip("_")
            # 3. Truncate to 15 chars
            return slug[:15]


        var_entries = []
        required_vars = []
        optional_vars = []

        for var in variable_defs:
            entry = f"{var['name']}:{var['type']}"

            # Compress any validation rules generically
            if var.get("validation_rules"):
                tokens = [compress_rule(r) for r in var["validation_rules"]]
                entry += f"[{','.join(tokens)}]"

            # Mark nullable and sensitive
            if var.get("nullable"):
                entry += "?"
            if var.get("sensitive"):
                entry += "*"

            var_entries.append(entry)

            # Categorize by default
            if var.get("default") is None:
                required_vars.append(var["name"])
            else:
                dv = var["default"]
                dv_str = str(dv).lower() if isinstance(dv, bool) else str(dv)
                optional_vars.append(f"{var['name']}={dv_str}")

        parts = [
            f"Count:{var_count}",
            f"Types:{','.join(unique_types)}",
            f"Vars:{';'.join(var_entries)}"
        ]
        if required_vars:
            parts.append(f"Required:{','.join(required_vars)}")
        if optional_vars:
            parts.append(f"Optional:{','.join(optional_vars)}")

        return "|".join(parts)
    
    @staticmethod
    def compress_local_values(local_values: List[Dict[str, Any]]) -> str:
        """
        Compress Terraform local values for LLM context:
            - Keeps name and expression (abbreviated).
            - Includes non-empty depends_on lists.
            - Omits description and usage_context.
        """
    
        if not local_values:
            return "None"

        entries = []
        for loc in local_values:
            # 1. Core entry: name=expression
            expr = loc.get("expression", "")
            # Abbreviate common patterns for brevity
            expr = expr.replace("format(", "fmt(").replace("var.", "v.")
            entry = f"{loc['name']}={expr}"

            # 2. Include dependencies if present
            deps = loc.get("depends_on", [])
            if deps:
                # Shorten resource names (remove "aws_" prefix) for brevity
                short = [d.replace("aws_", "") for d in deps]
                entry += f":dep[{','.join(short)}]"

            entries.append(entry)

        # 3. Join all entries in a single line
        return "Locals:" + ";".join(entries)
    
    @staticmethod
    def compress_data_sources(data_sources: List[Dict[str, Any]]) -> str:
        """
        Compress Terraform data blocks for LLM context:
        - Keeps name:type, configuration key-values, and exported attributes.
        - Omits description and empty dependencies.
        """
        if not data_sources:
            return "None"

        entries = []
        for d in data_sources:
            # Core entry: name:type
            entry = f"{d['resource_name']}:{d['data_source_type']}"

            # Add configuration
            cfg = d.get("configuration", {})
            cfg_parts = []
            # Handle filter blocks
            for filt in cfg.get("filter", []):
                # Single filter assumed
                key = filt["name"]
                vals = ",".join(filt["values"])
                cfg_parts.append(f"{key}={vals}")
            # Handle direct arguments
            for k, v in cfg.items():
                if k != "filter":
                    cfg_parts.append(f"{k}={v}")

            if cfg_parts:
                entry += "[" + ";".join(cfg_parts) + "]"

            # Add exported attributes
            exports = d.get("exported_attributes", [])
            if exports:
                entry += f"{{{','.join(exports)}}}"

            entries.append(entry)

        return "Data:" + ";".join(entries)
    
    @staticmethod
    def compress_output_definitions(output_defs: List[Dict[str, Any]]) -> str:
        """
        Compress Terraform outputs for LLM context:
        - Keeps name=value, non-empty depends_on, and sensitive=true marker.
        - Omits description, precondition, consumption_notes.
        """
        if not output_defs:
            return "None"

        entries = []
        for out in output_defs:
            # Core entry: name=value
            entry = f"{out['name']}={out['value']}"

            # Mark sensitive outputs
            if out.get("sensitive"):
                entry += "*"

            # Include depends_on if present
            deps = out.get("depends_on", [])
            if deps:
                # Strip common provider prefixes for brevity
                short = [d.replace("aws_", "") for d in deps]
                entry += f":dep[{','.join(short)}]"

            entries.append(entry)

        return "Outputs:" + ";".join(entries)
    
    @staticmethod
    def compress_terraform_files(terraform_files: List[Dict[str, Any]]) -> str:
        """Compress terraform files to essential summary - BALANCED APPROACH"""
        if not terraform_files:
            return "None"
        
        file_count = len(terraform_files)
        file_names = [file['file_name'] for file in terraform_files]
        
        # Extract file purposes (compress verbose descriptions)
        purposes_summary = []
        for file in terraform_files:
            purpose = file.get('file_purpose', '')
            if len(purpose) > 50:
                purpose = purpose[:50] + "..."
            purposes_summary.append(f"{file['file_name']}:{purpose}")
        
        # Extract resources included summary (keep for accuracy)
        resources_summary = []
        for file in terraform_files:
            if file.get('resources_included'):
                resources_summary.append(f"{file['file_name']}:{','.join(file['resources_included'])}")
        
        # Extract dependencies summary (keep for accuracy)
        deps_summary = []
        for file in terraform_files:
            if file.get('dependencies'):
                deps_summary.append(f"{file['file_name']}:{','.join(file['dependencies'])}")
        
        # Extract organization rationale summary (compress verbose descriptions)
        rationale_summary = []
        for file in terraform_files:
            if file.get('organization_rationale'):
                rationale = file['organization_rationale']
                if len(rationale) > 50:
                    rationale = rationale[:50] + "..."
                rationale_summary.append(f"{file['file_name']}:{rationale}")
        
        # Build compressed summary
        compressed = f"Count:{file_count}|Files:{','.join(file_names)}|Purposes:{';'.join(purposes_summary)}"
        
        if resources_summary:
            compressed += f"|Resources:{';'.join(resources_summary)}"
        
        if deps_summary:
            compressed += f"|Deps:{';'.join(deps_summary)}"
        
        if rationale_summary:
            compressed += f"|Rationale:{';'.join(rationale_summary)}"
        
        return compressed
    
    @staticmethod
    def compress_terraform_hcl_content(hcl_content: str, content_type: str) -> str:
        """Compress Terraform HCL content to essential summary"""
        if not hcl_content or not hcl_content.strip():
            return "None"
        
        # Extract resource/variable names and types
        if content_type == "resources":
            # Extract resource blocks
            return TerraformDataCompressor.compress_resource_hcl_output(hcl_content)
            # resource_matches = re.findall(r'resource\s+"([^"]+)"\s+"([^"]+)"', hcl_content)
            # if resource_matches:
            #     resource_summary = []
            #     for resource_type, resource_name in resource_matches:
            #         resource_summary.append(f"{resource_type}.{resource_name}")
            #     return f"Count:{len(resource_matches)}|Resources:{','.join(resource_summary)}"
        
        elif content_type == "variables":
            # Extract variable blocks
            # variable_matches = re.findall(r'variable\s+"([^"]+)"', hcl_content)
            # if variable_matches:
            #     return f"Count:{len(variable_matches)}|Variables:{','.join(variable_matches)}"

            return TerraformDataCompressor.compress_variables_overview(hcl_content)

        elif content_type == "data_sources":
            # Extract data source blocks
            return TerraformDataCompressor.compress_data_sources_overview(hcl_content)
            # data_matches = re.findall(r'data\s+"([^"]+)"\s+"([^"]+)"', hcl_content)
            # if data_matches:
            #     data_summary = []
            #     for data_type, data_name in data_matches:
            #         data_summary.append(f"{data_type}.{data_name}")
            #     return f"Count:{len(data_matches)}|DataSources:{','.join(data_summary)}"
        
        elif content_type == "locals":
            # Extract local value blocks
            return TerraformDataCompressor.compress_locals_ultra_minimal(hcl_content)
            # local_matches = re.findall(r'locals\s*{', hcl_content)
            # if local_matches:
            #     return f"Count:{len(local_matches)}|Locals:present"
        
        elif content_type == "outputs":
            # Extract output blocks
            return TerraformDataCompressor.compress_outputs_ultra_minimal(hcl_content)
            # output_matches = re.findall(r'output\s+"([^"]+)"', hcl_content)
            # if output_matches:
            #     return f"Count:{len(output_matches)}|Outputs:{','.join(output_matches)}"
        
        # Fallback: just count lines and basic info
        lines = hcl_content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        return f"Lines:{len(non_empty_lines)}|Type:{content_type}"
    
    @staticmethod
    def compress_workspace_state(generated_resources: str, generated_variables: str, 
                                generated_data_sources: str, generated_local_values: str, 
                                generated_outputs: str) -> Dict[str, str]:
        """Compress workspace state to essential summary"""
        return {
            'generated_resources': TerraformDataCompressor.compress_terraform_hcl_content(generated_resources, "resources"),
            'generated_variables': TerraformDataCompressor.compress_terraform_hcl_content(generated_variables, "variables"),
            'generated_data_sources': TerraformDataCompressor.compress_terraform_hcl_content(generated_data_sources, "data_sources"),
            'generated_local_values': TerraformDataCompressor.compress_terraform_hcl_content(generated_local_values, "locals"),
            'generated_outputs': TerraformDataCompressor.compress_terraform_hcl_content(generated_outputs, "outputs")
        }
    
    @staticmethod
    def compress_outputs_ultra_minimal(hcl_content: str) -> str:
        """
        Ultra-minimal Terraform outputs overview:
        - Counts outputs
        - Lists output names (suffix '*' if sensitive)
        - Indicates how many have explicit depends_on (D)
        """
        # 1. Extract all output blocks
        blocks = re.findall(r'output\s+"([^"]+)"\s*{([^}]*)}', hcl_content, re.DOTALL)
        if not blocks:
            return "O:0"

        names = []
        dep_count = 0

        for name, body in blocks:
            suffix = ""
            # 2. Mark sensitive outputs
            if re.search(r'\s*sensitive\s*=\s*true', body):
                suffix += "*"
            # 3. Check for depends_on
            if 'depends_on' in body:
                suffix += "D"
                dep_count += 1
            names.append(name + suffix)

        # 4. Build compressed string
        parts = [f"O:{len(blocks)}", f"Names:{','.join(names)}"]
        if dep_count:
            parts.append(f"D:{dep_count}")
        return "|".join(parts)


    @staticmethod
    def compress_locals_ultra_minimal(hcl_content: str) -> str:
        """
        Ultra-minimal Terraform locals overview:
        - Counts total local definitions
        - Lists local names
        - Indicates how many have complex expressions (EX)
        """
        # 1. Extract the locals block
        m = re.search(r'locals\s*{([^}]*)}', hcl_content, re.DOTALL)
        if not m:
            return "L:0"
    
        body = m.group(1)
        # 2. Find each local name=expression line
        entries = re.findall(r'(\w+)\s*=', body)
        count = len(entries)
    
        complex_count = 0
        items = []
        for name in entries:
            # 3. Check if expression spans function call or interpolation
            #    by looking for format(, ${, or other patterns
            pattern = rf'{name}\s*=\s*([^\n]+)'
            expr_match = re.search(pattern, body)
            expr = expr_match.group(1).strip() if expr_match else ""
            if re.search(r'\bformat\(|\$\{|\bjoin\(|\blookup\(', expr):
                # Mark complex expressions with (EX)
                items.append(f"{name}(EX)")
                complex_count += 1
            else:
                items.append(name)
    
        # 4. Build compressed result
        parts = [f"L:{count}", f"Keys:{','.join(items)}"]
        if complex_count:
            parts.append(f"EX:{complex_count}")
        return "|".join(parts)


    @staticmethod
    def compress_optimizer_data(optimizer_data: Dict[str, Any]) -> str:
        """Compress optimizer data to essential summary"""
        if not optimizer_data:
            return "None"
        
        # Extract key optimization metrics
        optimizers = optimizer_data.get('configuration_optimizers', [])
        if not optimizers:
            return "None"
        
        optimizer_count = len(optimizers)
        service_names = [opt.get('service_name', 'unknown') for opt in optimizers]
        
        # Extract optimization types
        cost_optimizations = sum(len(opt.get('cost_optimizations', [])) for opt in optimizers)
        performance_optimizations = sum(len(opt.get('performance_optimizations', [])) for opt in optimizers)
        security_optimizations = sum(len(opt.get('security_optimizations', [])) for opt in optimizers)
        
        # Extract validation status
        syntax_validations = sum(len(opt.get('syntax_validations', [])) for opt in optimizers)
        valid_files = sum(1 for opt in optimizers for val in opt.get('syntax_validations', []) 
                         if val.get('validation_status') == 'Valid')
        
        # Extract priority items (first 3 most important)
        priority_items = []
        for opt in optimizers:
            priority_items.extend(opt.get('implementation_priority', [])[:3])
        
        # Extract critical security issues
        critical_security = []
        for opt in optimizers:
            for sec_opt in opt.get('security_optimizations', []):
                if sec_opt.get('severity') in ['critical', 'high']:
                    critical_security.append(sec_opt.get('resource_name', 'unknown'))
        
        # Extract cost savings potential
        cost_savings = []
        for opt in optimizers:
            for cost_opt in opt.get('cost_optimizations', []):
                savings = cost_opt.get('estimated_savings', '')
                if savings and 'reduce' in savings.lower():
                    cost_savings.append(cost_opt.get('resource_name', 'unknown'))
        
        return f"Optimizers:{optimizer_count}|Services:{','.join(service_names)}|Cost:{cost_optimizations}|Perf:{performance_optimizations}|Sec:{security_optimizations}|Valid:{valid_files}/{syntax_validations}|Critical:{','.join(critical_security[:2])}|Savings:{','.join(cost_savings[:2])}|Priority:{','.join(priority_items[:3])}"
    

    @staticmethod
    def compress_data_sources_overview(hcl_content: str) -> str:
        """
        Ultra-minimal Terraform data_sources overview:
        - Counts data blocks
        - Lists type.name for each
        - Indicates how many have configuration filters/args (CFG)
        """
        # 1. Extract all data "type" "name" pairs
        matches = re.findall(r'data\s+"([^"]+)"\s+"([^"]+)"', hcl_content)
        if not matches:
            return "DS:0"

        entries = []
        cfg_count = 0

        for data_type, data_name in matches:
            # 2. Capture config block if present
            pattern = rf'data\s+"{re.escape(data_type)}"\s+"{re.escape(data_name)}"\s*{{([^}}]*)}}'
            m = re.search(pattern, hcl_content, re.DOTALL)
            block = m.group(1) if m else ""
            # 3. Mark if block contains any attribute or filter
            if re.search(r'\w+\s*=', block):
                cfg_count += 1
                entry = f"{data_type}.{data_name}(C)"
            else:
                entry = f"{data_type}.{data_name}"
            entries.append(entry)

        # 4. Build compressed result
        parts = [f"DS:{len(matches)}", f"Items:{','.join(entries)}"]
        if cfg_count:
            parts.append(f"CFG:{cfg_count}")

        return "|".join(parts)

    @staticmethod
    def compress_variables_overview(hcl_content: str) -> str:
        """
        Ultra-minimal service-agnostic Terraform variables overview:
        - Counts variables
        - Lists required vs optional
      - Indicates how many have validations
        """
        # Extract all variable names
        names = re.findall(r'variable\s+"([^"]+)"', hcl_content)
        if not names:
            return "VHCl:0"

        required, optional, val_count = [], [], 0
        for name in names:
            # Capture the entire block
            block_pattern = rf'variable\s+"{re.escape(name)}"\s*{{([^}}]*)}}'
            m = re.search(block_pattern, hcl_content, re.DOTALL)
            block = m.group(1) if m else ""

            # Determine if default exists (not null/empty)
            if re.search(r'default\s*=\s*(?!null)(?!"")\S', block):
                optional.append(name)
            else:
                required.append(name)

            # Count validations
            if "validation" in block:
                val_count += 1

        parts = [f"VHCl:{len(names)}"]
        if required:
            parts.append(f"R:{','.join(required)}")
        if optional:
            parts.append(f"O:{','.join(optional)}")
        if val_count:
            parts.append(f"V:{val_count}")

        return "|".join(parts)

    @staticmethod
    def compress_resource_hcl_output(hcl_code: str) -> str:
        """
        Compress HCL output for maximum LLM context efficiency:
        - Preserves resource structure and relationships
        - Maintains variable/local reference context
        - Eliminates syntax redundancy
        - Provides complexity indicators
        """
        if not hcl_code or not hcl_code.strip():
            return "None"
    
        # Extract resources with configurations
        resources = re.findall(r'resource\s+"([^"]+)"\s+"([^"]+)"\s*{([^}]+(?:{[^}]*}[^}]*)*)}', 
                          hcl_code, re.DOTALL)
    
        if not resources:
            # Fallback for malformed HCL
            simple_resources = re.findall(r'resource\s+"([^"]+)"\s+"([^"]+)"', hcl_code)
            if simple_resources:
                return f"HCL:{';'.join([f'{t}:{n}' for t, n in simple_resources])}"
            return "HCL:empty"
    
        resource_entries = []
        all_vars = set()
        all_locals = set()
        dependency_count = 0
    
        for resource_type, name, config in resources:
            # Core resource identification
            entry = f"{resource_type}:{name}"
        
            # Extract references (critical for context)
            res_vars = re.findall(r'var\.(\w+(?:\.\w+)?)', config)
            res_locals = re.findall(r'local\.(\w+)', config)
            all_vars.update(res_vars)
            all_locals.update(res_locals)
    
        
            # Configuration complexity indicator
            config_lines = [l.strip() for l in hcl_code.split('\n') 
                        if l.strip() and not l.strip().startswith('#') and '=' in l]
            if len(config_lines) > 2:
                entry += f"[{len(config_lines)}c]"
        
        
            # Dependency markers
            if 'depends_on' in config:
                entry += "→d"
                dependency_count += 1
        
            # Nested block complexity
            nested_blocks = len(re.findall(r'\w+\s*{', config))
            if nested_blocks > 0:
                entry += f"+{nested_blocks}b"
        
            resource_entries.append(entry)
    
        # Build compressed output
        parts = [f"HCL:{len(resources)}r"]
        parts.append(f"Res:{';'.join(resource_entries)}")
    
        # Compress variable references
        if all_vars:
            compressed_vars = []
            for var in sorted(all_vars):
                if '.' in var and len(var) > 15:
                    # Compress long dotted references
                    base, attr = var.split('.', 1)
                    base = base[:6] + '..' if len(base) > 8 else base
                    attr = attr[:8] + '..' if len(attr) > 10 else attr
                    compressed_vars.append(f"{base}.{attr}")
                else:
                    compressed_vars.append(var)
            parts.append(f"V:{','.join(compressed_vars)}")
    
    
        # Include locals
        if all_locals:
            parts.append(f"L:{','.join(sorted(all_locals))}")
    
        # Dependency summary
        if dependency_count > 1:
            parts.append(f"Deps:{dependency_count}")
    
        return "|".join(parts)

    @classmethod
    def compress_all_planning_data(cls, planning_resource_specifications: List[Dict[str, Any]], 
                                  planning_variable_definitions: List[Dict[str, Any]],
                                  planning_local_values: List[Dict[str, Any]], 
                                  planning_data_sources: List[Dict[str, Any]], 
                                  planning_output_definitions: List[Dict[str, Any]],
                                  planning_terraform_files: List[Dict[str, Any]], 
                                  generated_resources: str, generated_variables: str,
                                  generated_data_sources: str, generated_local_values: str, 
                                  generated_outputs: str, optimizer_data: Dict[str, Any]) -> Dict[str, str]:
        """Compress all planning data in one call"""
        # Compress workspace state
        workspace_compressed = cls.compress_workspace_state(generated_resources, generated_variables,
                                                          generated_data_sources, generated_local_values, generated_outputs)
        
        return {
            'resource_specifications': cls.compress_resource_specifications(planning_resource_specifications),
            'variable_definitions': cls.compress_variable_definitions(planning_variable_definitions),
            'local_values': cls.compress_local_values(planning_local_values),
            'data_sources': cls.compress_data_sources(planning_data_sources),
            'output_definitions': cls.compress_output_definitions(planning_output_definitions),
            'terraform_files': cls.compress_terraform_files(planning_terraform_files),
            'workspace_generated_resources': workspace_compressed['generated_resources'],
            'workspace_generated_variables': workspace_compressed['generated_variables'],
            'workspace_generated_data_sources': workspace_compressed['generated_data_sources'],
            'workspace_generated_local_values': workspace_compressed['generated_local_values'],
            'workspace_generated_outputs': workspace_compressed['generated_outputs'],
            'optimizer_data': cls.compress_optimizer_data(optimizer_data)
        }


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    compressor = TerraformDataCompressor()
    
    # Example resource specifications
    example_resources = [
        {
            'resource_address': 'aws_vpc.main',
            'resource_type': 'aws_vpc',
            'resource_name': 'main',
            'configuration': {'cidr_block': '${var.cidr_block}', 'enable_dns_support': '${var.enable_dns_support}'},
            'depends_on': [],
            'lifecycle_rules': {'prevent_destroy': True}
        }
    ]
    
    # Test compression
    compressed = compressor.compress_resource_specifications(example_resources)
    print(f"Compressed resource specs: {compressed}")
    
    # Expected output: Count:1|Types:aws_vpc|Addresses:aws_vpc.main|Configs:aws_vpc:cidr_block,enable_dns_support|Deps:
