from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import UserProfileViewSet, ActivityLogViewSet, PaymentViewSet, NotificationViewSet

router = DefaultRouter()
router.register(r'users', UserProfileViewSet)
router.register(r'activity', ActivityLogViewSet)
router.register(r'payments', PaymentViewSet)
router.register(r'notifications', NotificationViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
