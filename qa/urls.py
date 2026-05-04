from django.urls import path

from . import views

urlpatterns = [
    path("answer/", views.OneShotAnswerView.as_view(), name="answer-one-shot"),
    path("documents/", views.DocumentUploadView.as_view(), name="document-upload"),
    path("documents/<uuid:document_id>/", views.DocumentDetailView.as_view(), name="document-detail"),
    path("documents/<uuid:document_id>/questions/", views.QuestionsView.as_view(), name="questions"),
    path("questions/", views.AllDocumentsQuestionsView.as_view(), name="questions-all"),
    path("reset/", views.ResetIndexView.as_view(), name="reset-index"),
]
