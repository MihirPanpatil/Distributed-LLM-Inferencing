from django.contrib import admin
from .models import WorkerNode, ModelShard, InferenceRequest

@admin.register(WorkerNode)
class WorkerNodeAdmin(admin.ModelAdmin):
    list_display = ('hostname', 'ip_address', 'port', 'is_active', 'last_heartbeat')
    search_fields = ('hostname', 'ip_address')

@admin.register(ModelShard)
class ModelShardAdmin(admin.ModelAdmin):
    list_display = ('model_name', 'shard_id', 'node', 'is_loaded')
    list_filter = ('model_name', 'is_loaded')
    search_fields = ('model_name',)

@admin.register(InferenceRequest)
class InferenceRequestAdmin(admin.ModelAdmin):
    list_display = ('model_name', 'status', 'created_at', 'completed_at')
    list_filter = ('status', 'model_name')
    search_fields = ('model_name', 'prompt')
