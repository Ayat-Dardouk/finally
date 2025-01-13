from django.shortcuts import render, redirect, get_object_or_404
from django.core.files.storage import FileSystemStorage
from django.views import View
from django.views.generic import ListView, DetailView
from .models import JobPosition, Application, Company

from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.http import HttpResponseForbidden
from django.contrib.auth import views as auth_views

from django.contrib.auth import logout
from django.shortcuts import redirect
from .utils import match_cvs_to_job
import joblib
from django.utils import timezone

from django.contrib import messages

def logout_view(request):
    logout(request)
    return redirect('login')

from django.shortcuts import get_object_or_404, redirect
from django.views import View
from django.urls import reverse_lazy
from .models import JobPosition

class DeleteJobPositionView(View):
    def post(self, request, pk):
        job_position = get_object_or_404(JobPosition, pk=pk)
        job_position.delete()
        return redirect('list_job_positions')  # Redirect to the job positions list after deletion

class LoginView(auth_views.LoginView):
    template_name = 'jobs/login.html'




class ListJobPositionsView(ListView):
    model = JobPosition
    template_name = 'jobs/list_job_positions.html'
    context_object_name = 'job_positions'

    def get_queryset(self):
        queryset = super().get_queryset()
        company_id = self.request.GET.get('company')
        if company_id:
            queryset = queryset.filter(company_id=company_id)
        return queryset

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['companies'] = Company.objects.all()
        return context


class UploadCVView(View):
    def get(self, request):
        return render(request, 'upload_cv.html')

    def post(self, request):
        applicant_name = request.POST['applicant_name']
        applicant_email = request.POST['applicant_email']
        job_position_id = request.POST['job_position_id']
        cv_file = request.FILES['cv']
        cover_letter = request.POST.get('cover_letter', '')

        job_position = get_object_or_404(JobPosition, id=job_position_id)

        # Check if an application with this email already exists for this job
        if Application.objects.filter(job_position=job_position, applicant_email=applicant_email).exists():
            messages.error(request, "You have already applied for this job position.")
            return redirect('upload_cv')  # Replace with your desired redirect or render.

        fs = FileSystemStorage()
        filename = fs.save(cv_file.name, cv_file)
        file_path = fs.path(filename)

        with open(file_path, 'r') as f:
            extracted_text = f.read()

        Application.objects.create(
            job_position=job_position,
            applicant_name=applicant_name,
            applicant_email=applicant_email,
            cv=filename,
            cover_letter=cover_letter,
            extracted_text=extracted_text
        )
        messages.success(request, "Your application has been submitted successfully.")
        return redirect('list_applications')


class ListApplicationsView(ListView):
    model = Application
    template_name = 'list_applications.html'
    context_object_name = 'applications'

class ViewJobPositionView(DetailView):
    model = JobPosition
    template_name = 'jobs/view_job_position.html'
    context_object_name = 'job_position'
    pk_url_kwarg = 'job_id'

class SubmitApplicationView(View):
    def get(self, request, job_id):
        job_position = get_object_or_404(JobPosition, id=job_id)
        return render(request, 'jobs/submit_application.html', {'job_position': job_position})

    def post(self, request, job_id):
        if request.FILES['cv']:
            cv_file = request.FILES['cv']
            applicant_name = request.POST['applicant_name']
            applicant_email = request.POST['applicant_email']
            cover_letter = request.POST.get('cover_letter', '')

            job_position = get_object_or_404(JobPosition, id=job_id)

            # Check if an application with this email already exists for this job
            if Application.objects.filter(job_position=job_position, applicant_email=applicant_email).exists():
                messages.error(request, "You have already applied for this job position.")
                return redirect('submit_application', job_id=job_id)

            fs = FileSystemStorage()
            filename = fs.save(cv_file.name, cv_file)

            Application.objects.create(
                job_position=job_position,
                applicant_name=applicant_name,
                applicant_email=applicant_email,
                cv=filename,
                cover_letter=cover_letter
            )
            messages.success(request, "Your application has been submitted successfully.")

        return redirect('view_job_position', job_id=job_id)
    

from django.shortcuts import render, get_object_or_404, redirect
from django.views import View
from .models import Company, JobPosition

from django.utils import timezone

from django.shortcuts import render, get_object_or_404, redirect
from django.utils import timezone
from .models import JobPosition, Company

from django.shortcuts import render, get_object_or_404, redirect
from django.utils import timezone
from .models import JobPosition, Company

from datetime import datetime

class AddJobPositionView(View):
    def get(self, request, company_id):
        company = get_object_or_404(Company, id=company_id)
        
        # Ensure the user is the admin of the company
        if company.admin != request.user:
            return redirect('some_error_page')  # Redirect to an error page or permission error page
        
        # Render the job position form
        return render(request, 'jobs/add_job_position.html', {'company': company})

    def post(self, request, company_id):
        company = get_object_or_404(Company, id=company_id)

        # Ensure the user is the admin of the company
        if company.admin != request.user:
            return redirect('some_error_page')  # Redirect to an error page or permission error page

        # Get the form data
        title = request.POST.get('title')
        description = request.POST.get('description')
        required_skills = request.POST.get('required_skills', '')
        application_deadline = request.POST.get('application_deadline')
        salary_range = request.POST.get('salary_range', '')
        job_type = request.POST.get('job_type')

    

        # Create and save the new job position
        job_position = JobPosition(
            title=title,
            company=company,
            description=description,
            required_skills=required_skills,
            application_deadline=application_deadline,
            salary_range=salary_range,
            job_type=job_type,
        )
        job_position.save()

        # Redirect to the view page for the job position
        return redirect('view_job_position', job_id=job_position.id)



class JobAppliedApplicationsView(View):
    @method_decorator(login_required)
    def get(self, request, job_id):
        job_position = get_object_or_404(JobPosition, id=job_id)
        if request.user != job_position.company.admin:
            return HttpResponseForbidden("You are not allowed to view these applications.")
        applications = Application.objects.filter(job_position=job_position)
        return render(request, 'jobs/matched_applications.html', {'job_position': job_position, 'applications': applications})
    

class UserCompanyPositionsView(ListView):
    model = JobPosition
    template_name = 'jobs/user_company_positions.html'
    context_object_name = 'job_positions'

    @method_decorator(login_required)
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)

    def get_queryset(self):
        queryset = super().get_queryset()
        if self.request.user.is_authenticated:
            # Assuming the Company model has a foreign key to User model as 'admin'
            queryset = queryset.filter(company__admin=self.request.user)
        return queryset

    def get_context_data(self, **kwargs):
       context = super().get_context_data(**kwargs)
       if self.request.user.is_authenticated:
        # Fetch all companies associated with the user
        companies = Company.objects.filter(admin=self.request.user)
        context['companies'] = companies
       return context


class JobMatchingCVsView(View):
    @method_decorator(login_required)
    def get(self, request, job_id):
        job_position = get_object_or_404(JobPosition, id=job_id)
        if request.user != job_position.company.admin:
            return HttpResponseForbidden("You are not allowed to view these applications.")
        
        # Load the trained model and vectorizer
        model = joblib.load('jobs/cv_classifier.pkl')  # Load the saved model
        vectorizer = joblib.load('jobs/tfidf_vectorizer.pkl')  # Load the saved vectorizer

        job_description = job_position.description
        matched_applications, scores = match_cvs_to_job(job_description, job_position, model, vectorizer)

        if not matched_applications:
            # If no applicants have applied, show an error message
            messages.error(request, "No applicants have applied for this job position.")

        return render(request, 'jobs/matched_cvs.html', {
            'job_position': job_position,
            'matched_applications': matched_applications,
            'scores': scores
        })


from django import template

register = template.Library()

@register.filter
def break_filter(value):
    """Simulate a break in a for loop."""
    return None

@register.filter
def continue_filter(value):
    """Simulate a continue in a for loop."""
    return value
