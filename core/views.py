from rest_framework import viewsets
from .models import (
    UserProfile, Content, ActivityLog, Payment, Notification, 
    ParentControl, DevelopmentMilestone, PersonalizedAdvice, 
    FriendRequest, Message
)
from .serializers import (
    UserProfileSerializer, ContentSerializer, ActivityLogSerializer, PaymentSerializer, 
    NotificationSerializer, ParentControlSerializer, DevelopmentMilestoneSerializer, 
    PersonalizedAdviceSerializer, FriendRequestSerializer, MessageSerializer
)

class UserProfileViewSet(viewsets.ModelViewSet):
    queryset = UserProfile.objects.all()
    serializer_class = UserProfileSerializer


class ContentViewSet(viewsets.ModelViewSet):
    queryset = Content.objects.all()
    serializer_class = ContentSerializer


class ActivityLogViewSet(viewsets.ModelViewSet):
    queryset = ActivityLog.objects.all()
    serializer_class = ActivityLogSerializer


class PaymentViewSet(viewsets.ModelViewSet):
    queryset = Payment.objects.all()
    serializer_class = PaymentSerializer


class NotificationViewSet(viewsets.ModelViewSet):
    queryset = Notification.objects.all()
    serializer_class = NotificationSerializer


class ParentControlViewSet(viewsets.ModelViewSet):
    queryset = ParentControl.objects.all()
    serializer_class = ParentControlSerializer


class DevelopmentMilestoneViewSet(viewsets.ModelViewSet):
    queryset = DevelopmentMilestone.objects.all()
    serializer_class = DevelopmentMilestoneSerializer


class PersonalizedAdviceViewSet(viewsets.ModelViewSet):
    queryset = PersonalizedAdvice.objects.all()
    serializer_class = PersonalizedAdviceSerializer


class FriendRequestViewSet(viewsets.ModelViewSet):
    queryset = FriendRequest.objects.all()
    serializer_class = FriendRequestSerializer


class MessageViewSet(viewsets.ModelViewSet):
    queryset = Message.objects.all()
    serializer_class = MessageSerializer
