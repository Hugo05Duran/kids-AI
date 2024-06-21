from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    UserProfileViewSet, ContentViewSet, ActivityLogViewSet, PaymentViewSet, 
    NotificationViewSet, ParentControlViewSet, DevelopmentMilestoneViewSet, 
    PersonalizedAdviceViewSet, FriendRequestViewSet, MessageViewSet
)

router = DefaultRouter()
router.register(r'users', UserProfileViewSet)
router.register(r'content', ContentViewSet)
router.register(r'activity', ActivityLogViewSet)
router.register(r'payments', PaymentViewSet)
router.register(r'notifications', NotificationViewSet)
router.register(r'parentcontrols', ParentControlViewSet)
router.register(r'milestones', DevelopmentMilestoneViewSet)
router.register(r'advices', PersonalizedAdviceViewSet)
router.register(r'friendrequests', FriendRequestViewSet)
router.register(r'messages', MessageViewSet)

urlpatterns = [
    path('', include(router.urls)),
]