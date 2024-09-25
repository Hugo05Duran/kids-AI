from django.db import models
from django.contrib.auth.models import AbstractUser


class UserProfile(AbstractUser):
    age = models.IntegerField(null=True, blank=True)
    bio = models.TextField(null=True, blank=True)

    def __str__(self):
        return self.username


class ActivityLog(models.Model):
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    action = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.action}"


class Payment(models.Model):
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    payment_date = models.DateTimeField(auto_now_add=True)
    subscription_type = models.CharField(max_length=255)

    def __str__(self):
        return f"{self.user.username} - {self.subscription_type} - {self.amount}"


class Notification(models.Model):
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    message = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    read = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.user.username} - {self.message}"
