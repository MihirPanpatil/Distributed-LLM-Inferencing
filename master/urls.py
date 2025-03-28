from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('dashboard.urls', namespace='dashboard')),
    path('shard/', include('shard_mgmt.urls', namespace='shard_mgmt')),
]
