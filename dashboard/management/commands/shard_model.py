import os
import json
import torch
import argparse
from django.core.management.base import BaseCommand
from transformers import AutoModelForCausalLM, AutoTokenizer
from shard_mgmt.models import ModelShard, WorkerNode

class Command(BaseCommand):
    help = 'Split a Hugging Face model into shards for distributed inference'

    def add_arguments(self, parser):
        parser.add_argument('--model_name', type=str, required=True, help='Hugging Face model name')
        parser.add_argument('--num_shards', type=int, required=True, help='Number of shards to create')
        parser.add_argument('--output_dir', type=str, default='model_shards', help='Output directory for shards')
        parser.add_argument('--device', type=str, default='cpu', help='Device to use for model loading (cpu or cuda)')
        parser.add_argument('--register_db', action='store_true', help='Register shards in database')
        parser.add_argument('--node_id', type=int, help='Worker node ID to associate with shards if registering in DB')

    def handle(self, *args, **options):
        model_name = options['model_name']
        num_shards = options['num_shards']
        output_dir = options['output_dir']
        device = options['device']
        register_db = options['register_db']
        node_id = options['node_id']
        
        self.stdout.write(self.style.SUCCESS(f'Sharding model {model_name} into {num_shards} shards on {device}'))
        
        # Validate device
        if device not in ['cpu', 'cuda']:
            self.stdout.write(self.style.ERROR(f'Invalid device: {device}. Use "cpu" or "cuda".'))
            return
        
        if device == 'cuda' and not torch.cuda.is_available():
            self.stdout.write(self.style.WARNING('CUDA requested but not available. Falling back to CPU.'))
            device = 'cpu'
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        model_output_dir = os.path.join(output_dir, model_name.replace('/', '_'))
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Load tokenizer
        self.stdout.write('Loading tokenizer...')
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer_path = os.path.join(model_output_dir, 'tokenizer')
            os.makedirs(tokenizer_path, exist_ok=True)
            tokenizer.save_pretrained(tokenizer_path)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Failed to load tokenizer: {str(e)}'))
            return
        
        # Load model
        self.stdout.write('Loading model...')
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Failed to load model: {str(e)}'))
            return
        
        # Get model layers
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            # GPT-2 style models
            layers = model.transformer.h
            num_layers = len(layers)
            model_type = 'gpt2'
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # OPT style models
            layers = model.model.layers
            num_layers = len(layers)
            model_type = 'opt'
        elif hasattr(model, 'layers'):
            # Some models have layers at the top level
            layers = model.layers
            num_layers = len(layers)
            model_type = 'generic'
        else:
            self.stdout.write(self.style.ERROR('Unsupported model architecture'))
            return
        
        self.stdout.write(f'Model has {num_layers} layers (type: {model_type})')
        
        # Calculate layers per shard
        layers_per_shard = num_layers // num_shards
        remainder = num_layers % num_shards
        
        if layers_per_shard == 0:
            self.stdout.write(self.style.ERROR(f'Too many shards ({num_shards}) for model with {num_layers} layers'))
            return
        
        # Prepare for DB registration if requested
        node = None
        if register_db and node_id:
            try:
                node = WorkerNode.objects.get(id=node_id)
                self.stdout.write(f'Will register shards with node: {node.hostname}')
            except WorkerNode.DoesNotExist:
                self.stdout.write(self.style.ERROR(f'Worker node with ID {node_id} not found'))
                register_db = False
        
        # Create shards
        created_shards = []
        for shard_id in range(num_shards):
            self.stdout.write(f'Creating shard {shard_id}...')
            
            # Calculate start and end layers for this shard
            start_layer = shard_id * layers_per_shard
            end_layer = start_layer + layers_per_shard
            if shard_id == num_shards - 1:
                end_layer += remainder
            
            self.stdout.write(f'Shard {shard_id} will contain layers {start_layer} to {end_layer-1}')
            
            # Create a new model with only the layers for this shard
            shard_model = type(model)(model.config)
            
            # Copy embeddings and other components based on model type
            if model_type == 'gpt2':
                # GPT-2 style models
                shard_model.transformer.wte = model.transformer.wte
                shard_model.transformer.wpe = model.transformer.wpe
                shard_model.transformer.ln_f = model.transformer.ln_f
                
                # Copy only the layers for this shard
                for i in range(start_layer, end_layer):
                    shard_model.transformer.h[i] = model.transformer.h[i]
            
            elif model_type == 'opt':
                # OPT style models
                shard_model.model.embed_tokens = model.model.embed_tokens
                shard_model.model.embed_positions = model.model.embed_positions
                shard_model.model.final_layer_norm = model.model.final_layer_norm
                
                # Copy only the layers for this shard
                for i in range(start_layer, end_layer):
                    shard_model.model.layers[i] = model.model.layers[i]
            
            elif model_type == 'generic':
                # Generic model with top-level layers
                # This is a simplified approach and might need customization
                for i in range(start_layer, end_layer):
                    shard_model.layers[i] = model.layers[i]
            
            # Save shard
            shard_path = os.path.join(model_output_dir, f'shard_{shard_id}')
            os.makedirs(shard_path, exist_ok=True)
            
            try:
                shard_model.save_pretrained(shard_path)
                
                # Save metadata
                metadata = {
                    'model_name': model_name,
                    'shard_id': shard_id,
                    'num_shards': num_shards,
                    'start_layer': start_layer,
                    'end_layer': end_layer - 1,
                    'total_layers': num_layers,
                    'model_type': model_type
                }
                
                with open(os.path.join(shard_path, 'metadata.json'), 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Register in database if requested
                if register_db and node:
                    shard_obj, created = ModelShard.objects.update_or_create(
                        model_name=model_name,
                        shard_id=shard_id,
                        defaults={
                            'node': node,
                            'is_loaded': False,
                        }
                    )
                    action = 'Created' if created else 'Updated'
                    self.stdout.write(f'{action} database record for shard {shard_id}')
                    created_shards.append(shard_obj)
                
                self.stdout.write(self.style.SUCCESS(f'Shard {shard_id} created successfully'))
                
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Failed to save shard {shard_id}: {str(e)}'))
                continue
        
        self.stdout.write(self.style.SUCCESS(f'Model {model_name} sharded successfully into {num_shards} shards'))
        self.stdout.write(f'Shards are stored in {model_output_dir}')
        
        if register_db and created_shards:
            self.stdout.write(f'Registered {len(created_shards)} shards in the database')
