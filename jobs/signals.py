from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Application
from .utils import extract_text_from_pdf

@receiver(post_save, sender=Application)
def extract_text_from_cv(sender, instance, created, **kwargs):
    if created and instance.cv:
        cv_path = instance.cv.path
        extracted_text = extract_text_from_pdf(cv_path)
        instance.extracted_text = extracted_text
        instance.save()
