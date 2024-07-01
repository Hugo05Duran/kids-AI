from rest_framework import serializers
from .models import (
    UserProfile, Content, ActivityLog, Payment, Notification, 
    ParentControl, DevelopmentMilestone, PersonalizedAdvice, 
    FriendRequest, Message
)

class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserProfile
        fields = ['id', 'username', 'email', 'age', 'bio', 'groups', 'user_permissions']


class ContentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Content
        fields = ['id', 'title', 'description', 'content_type', 'url', 'created_at', 'updated_at']


class ActivityLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = ActivityLog
        fields = ['id', 'user', 'content', 'action', 'timestamp']


class PaymentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Payment
        fields = ['id', 'user', 'amount', 'payment_date', 'subscription_type']


class NotificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Notification
        fields = ['id', 'user', 'message', 'created_at', 'read']


class ParentControlSerializer(serializers.ModelSerializer):
    class Meta:
        model = ParentControl
        fields = ['id', 'user', 'content', 'restricted']


class DevelopmentMilestoneSerializer(serializers.ModelSerializer):
    class Meta:
        model = DevelopmentMilestone
        fields = ['id', 'user', 'milestone', 'achieved_date', 'notes']


class PersonalizedAdviceSerializer(serializers.ModelSerializer):
    class Meta:
        model = PersonalizedAdvice
        fields = ['id', 'user', 'advice', 'date_provided']


class FriendRequestSerializer(serializers.ModelSerializer):
    class Meta:
        model = FriendRequest
        fields = ['id', 'from_user', 'to_user', 'status', 'created_at']


class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ['id', 'sender', 'recipient', 'content', 'timestamp']
