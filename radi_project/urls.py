from django.contrib import admin
from django.urls import path, include               #n
from django.conf import settings                    #n
from django.conf.urls.static import static          #n  

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('radi_app.urls')),             #n
]

#n Fix this line - use MEDIA_ROOT instead of MEDIA_URL      
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)