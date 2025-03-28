from django.db import models
from django.utils import timezone

class WorkerNode(models.Model):
    hostname = models.CharField(max_length=255)
    ip_address = models.CharField(max_length=255)
    port = models.IntegerField(default=5000)
    is_active = models.BooleanField(default=False)
    last_heartbeat = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.hostname} ({self.ip_address}:{self.port})"
    
    def get_url(self):
        return f"http://{self.ip_address}:{self.port}"

class ModelShard(models.Model):
    node = models.ForeignKey(WorkerNode, on_delete=models.CASCADE, related_name='shards')
    model_name = models.CharField(max_length=255)
    shard_id = models.IntegerField()
    is_loaded = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ('model_name', 'shard_id')
    
    def __str__(self):
        return f"{self.model_name} - Shard {self.shard_id} on {self.node.hostname}"

class InferenceRequest(models.Model):
    STATUS_CHOICES = (
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    )
    
    model_name = models.CharField(max_length=255)
    prompt = models.TextField()
    result = models.TextField(null=True, blank=True)
    error = models.TextField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.model_name} - {self.status} - {self.created_at}"
    
    def mark_completed(self, result):
        self.result = result
        self.status = 'completed'
        self.completed_at = timezone.now()
        self.save()
    
    def mark_failed(self, error):
        self.error = error
        self.status = 'failed'
        self.completed_at = timezone.now()
        self.save()
