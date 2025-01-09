from django.urls import path
from .views import (
    ListJobPositionsView, 
    UploadCVView, 
    ListApplicationsView, 
    ViewJobPositionView, 
    SubmitApplicationView, 
    JobAppliedApplicationsView, 
    LoginView, 
    logout_view, 
    UserCompanyPositionsView, 
    JobMatchingCVsView, 
    AddJobPositionView
)

urlpatterns = [
    path('jobs/', ListJobPositionsView.as_view(), name='list_job_positions'),
    path('upload_cv/', UploadCVView.as_view(), name='upload_cv'),
    path('applications/', ListApplicationsView.as_view(), name='list_applications'),
    path('jobs/<int:job_id>/', ViewJobPositionView.as_view(), name='view_job_position'),
    path('jobs/<int:job_id>/submit_application/', SubmitApplicationView.as_view(), name='submit_application'),
    path('jobs/<int:job_id>/applied_applications/', JobAppliedApplicationsView.as_view(), name='job_applied_applications'),
    path('jobs/<int:job_id>/matched_cvs/', JobMatchingCVsView.as_view(), name='job_matching_cvs'),
    path('login/', LoginView.as_view(), name='login'),
    path('logout/', logout_view, name='logout'),
    path('my_company_positions/', UserCompanyPositionsView.as_view(), name='user_company_positions'),
    path('add_job_position/<int:company_id>/', AddJobPositionView.as_view(), name='add_job_position'),
]
