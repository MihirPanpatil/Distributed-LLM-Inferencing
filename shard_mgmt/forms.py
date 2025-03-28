from django import forms
from .models import WorkerNode, InferenceRequest

class WorkerNodeForm(forms.ModelForm):
    class Meta:
        model = WorkerNode
        fields = ['hostname', 'ip_address', 'port']
        widgets = {
            'hostname': forms.TextInput(attrs={'class': 'form-control'}),
            'ip_address': forms.TextInput(attrs={'class': 'form-control'}),
            'port': forms.NumberInput(attrs={'class': 'form-control'}),
        }

class InferenceForm(forms.ModelForm):
    class Meta:
        model = InferenceRequest
        fields = ['model_name', 'prompt']
        widgets = {
            'model_name': forms.TextInput(attrs={'class': 'form-control'}),
            'prompt': forms.Textarea(attrs={'class': 'form-control', 'rows': 5}),
        }
