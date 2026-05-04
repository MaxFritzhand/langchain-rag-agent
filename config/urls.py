from django.conf import settings
from django.conf.urls.static import static
from django.urls import include, path

from qa.views import index_view

urlpatterns = [
    path("", index_view, name="index"),
    path("api/", include("qa.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
