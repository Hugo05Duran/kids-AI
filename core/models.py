from django.db import models
from django.contrib.auth.models import AbstractUser, Group, Permission


class UserProfile(AbstractUser):
    
    age = models.IntegerField(null=True, blank=True)
    bio = models.TextField(null=True, blank=True)

    groups = models.ManyToManyField(
        Group,
        related_name='user_profile_set',
        blank=True,
        help_text='The groups this user belongs to.',
        verbose_name='groups',
    )
    
    user_permissions = models.ManyToManyField(
        Permission,
        related_name='user_profile_set',
        blank=True,
        help_text='Specific permissions for this user.',
        verbose_name='user permissions',
    )


class Content(models.Model):
    CONTENT_TYPE_CHOICES = [
        ('video', 'Video'),
        ('story', 'Story'),
        ('game', 'Game'),
    ]

    title = models.CharField(max_length=255)
    description = models.TextField()
    content_type = models.CharField(max_length=20, choices=CONTENT_TYPE_CHOICES)
    url = models.URLField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title


class ActivityLog(models.Model):
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    content = models.ForeignKey(Content, on_delete=models.CASCADE)
    action = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.action} - {self.content.title}"


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


class ParentControl(models.Model):
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    content = models.ForeignKey(Content, on_delete=models.CASCADE)
    restricted = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.user.username} - {self.content.title} - {'Restricted' if self.restricted else 'Allowed'}"


class DevelopmentMilestone(models.Model):
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    milestone = models.CharField(max_length=255)
    achieved_date = models.DateField()
    notes = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"{self.user.username} - {self.milestone} - {self.achieved_date}"


class PersonalizedAdvice(models.Model):
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    advice = models.TextField()
    date_provided = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.date_provided}"


class FriendRequest(models.Model):
    from_user = models.ForeignKey(UserProfile, related_name='from_user', on_delete=models.CASCADE)
    to_user = models.ForeignKey(UserProfile, related_name='to_user', on_delete=models.CASCADE)
    status = models.CharField(max_length=20, choices=[('pending', 'Pending'), ('accepted', 'Accepted'), ('rejected', 'Rejected')])
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.from_user.username} to {self.to_user.username} - {self.status}"


class Message(models.Model):
    sender = models.ForeignKey(UserProfile, related_name='sent_messages', on_delete=models.CASCADE)
    recipient = models.ForeignKey(UserProfile, related_name='received_messages', on_delete=models.CASCADE)
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"From {self.sender.username} to {self.recipient.username} - {self.timestamp}"