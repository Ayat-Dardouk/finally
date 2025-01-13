from django.db import models
from django_extensions.db.models import TimeStampedModel
from django.contrib.auth.models import User
from django.utils import timezone


class Company(TimeStampedModel):
    name = models.CharField(max_length=255)
    location = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    website = models.URLField(blank=True, null=True)
    established_date = models.DateField(blank=True, null=True)
    admin = models.ForeignKey(User, on_delete=models.CASCADE, related_name='companies')

    def __str__(self):
        return self.name


class JobPosition(TimeStampedModel):
    company = models.ForeignKey(Company, on_delete=models.CASCADE, related_name='job_positions')
    title = models.CharField(max_length=255)
    description = models.TextField()
    required_skills = models.TextField()  # Comma-separated skills
    application_deadline = models.DateField(blank=True, null=True)
    salary_range = models.CharField(max_length=100, blank=True, null=True)
    job_type = models.CharField(
        max_length=50,
        choices=[('FT', 'Full-Time'), ('PT', 'Part-Time'), ('CT', 'Contract')]
    )

    def __str__(self):
        return f"{self.title} at {self.company.name}"

class Application(TimeStampedModel):
    job_position = models.ForeignKey(JobPosition, on_delete=models.CASCADE, related_name='applications')
    applicant_name = models.CharField(max_length=255)
    applicant_email = models.EmailField()
    cv = models.FileField(upload_to='cvs/')
    cover_letter = models.TextField(blank=True, null=True)
    status = models.CharField(max_length=50, choices=[('P', 'Pending'), ('R', 'Reviewed'), ('A', 'Accepted'), ('D', 'Declined')], default='P')
    extracted_text = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"Application by {self.applicant_name} for {self.job_position.title}"