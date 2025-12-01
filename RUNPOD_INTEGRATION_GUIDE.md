# RunPod RAG + EC2 Django 서버 통합 가이드

**작성일:** 2025-12-01
**목적:** RunPod의 RAG 시스템을 EC2 Django 챗봇과 연결

---

## 📋 목차

1. [아키텍처 개요](#아키텍처-개요)
2. [RunPod 설정](#runpod-설정)
3. [EC2 Django 서버 리팩토링](#ec2-django-서버-리팩토링)
4. [배포 및 테스트](#배포-및-테스트)
5. [문제 해결](#문제-해결)

---

## 🏗️ 아키텍처 개요

### 전체 흐름

```
사용자 브라우저
    ↓ (React)
EC2 - React Frontend (Port 3000)
    ↓ (Axios HTTP Request)
EC2 - Django Backend (Port 8000)
    ↓ (HTTPS Request)
RunPod - FastAPI RAG Server (Port 8080)
    ↓
LangGraph RAG / Optimized RAG
    ↓ (Response with answer + sources)
EC2 - Django Backend
    ↓ (Save to MySQL)
EC2 - MySQL Database
    ↓ (Return response)
사용자 브라우저 (Display answer)
```

### 주요 컴포넌트

| 컴포넌트 | 위치 | 역할 |
|---------|------|------|
| **React Frontend** | EC2 | 사용자 인터페이스 |
| **Django Backend** | EC2 | 비즈니스 로직, 인증, DB 관리 |
| **FastAPI RAG Server** | RunPod | RAG 처리 (LangGraph/Optimized) |
| **MySQL** | EC2 | 채팅 내역, 사용자 정보 저장 |
| **ChromaDB** | RunPod | 벡터 검색 DB |

---

## 🚀 RunPod 설정

### 1. FastAPI 서버 실행

```bash
cd /workspace/testrag/experiments/rag_pipeline

# LangGraph RAG 사용 (고품질)
python serve_unified.py --rag-type langgraph --port 8080

# 또는 Optimized RAG 사용 (빠른 응답)
python serve_unified.py --rag-type optimized --port 8080
```

**예상 출력:**
```
2025-12-01 10:00:00 - INFO - Starting Unified RAG API Server
2025-12-01 10:00:00 - INFO - RAG Type: langgraph
2025-12-01 10:00:00 - INFO - Config: config/enhanced.yaml
2025-12-01 10:00:05 - INFO - LangGraph RAG loaded in 5.23s
2025-12-01 10:00:05 - INFO - Server: http://0.0.0.0:8080
2025-12-01 10:00:05 - INFO - API Docs: http://0.0.0.0:8080/docs
INFO:     Uvicorn running on http://0.0.0.0:8080
```

### 2. RunPod 포트 포워딩

RunPod 대시보드에서:
1. Pod 설정 → **Ports**
2. **8080** 포트 추가
3. Public URL 확인 (예: `https://xxxxx-8080.proxy.runpod.net`)
4. URL 복사 → Django 서버 환경변수로 사용

### 3. Health Check

```bash
# RunPod에서 직접 테스트
curl http://localhost:8080/api/v1/health

# 예상 출력:
{
  "status": "healthy",
  "rag_type": "langgraph",
  "rag_loaded": true,
  "message": "RAG system ready"
}
```

### 4. 백그라운드 실행 (nohup)

```bash
nohup python serve_unified.py --rag-type langgraph --port 8080 > server.log 2>&1 &

# 로그 확인
tail -f server.log

# 프로세스 확인
ps aux | grep serve_unified

# 중지
pkill -f serve_unified
```

---

## 🔧 EC2 Django 서버 리팩토링

### 1. Django 모델 추가

**파일 위치:** `backend/apps/chatbot/models.py`

```python
from django.db import models
from django.conf import settings


class ChatSession(models.Model):
    """채팅 세션"""
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='chat_sessions'
    )
    title = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'chat_sessions'
        ordering = ['-updated_at']
        indexes = [
            models.Index(fields=['-updated_at']),
            models.Index(fields=['user', '-updated_at']),
        ]

    def __str__(self):
        return f"{self.user.username} - {self.title}"


class ChatMessage(models.Model):
    """채팅 메시지"""
    ROLE_CHOICES = (
        ('user', '사용자'),
        ('assistant', 'AI'),
    )

    session = models.ForeignKey(
        ChatSession,
        on_delete=models.CASCADE,
        related_name='messages'
    )
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    sources = models.JSONField(default=list, blank=True)  # RAG 참고 문서

    # 메타데이터
    metadata = models.JSONField(default=dict, blank=True)  # RAG 타입, 응답 시간 등

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'chat_messages'
        ordering = ['created_at']
        indexes = [
            models.Index(fields=['session', 'created_at']),
        ]

    def __str__(self):
        return f"{self.role}: {self.content[:50]}"


class ChatBookmark(models.Model):
    """채팅 북마크"""
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='chat_bookmarks'
    )
    message = models.ForeignKey(
        ChatMessage,
        on_delete=models.CASCADE,
        null=True,
        blank=True
    )
    content = models.TextField()
    sources = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'chat_bookmarks'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', '-created_at']),
        ]

    def __str__(self):
        return f"{self.user.username} - {self.content[:50]}"
```

### 2. Serializers 추가

**파일 위치:** `backend/apps/chatbot/serializers.py`

```python
from rest_framework import serializers
from .models import ChatSession, ChatMessage, ChatBookmark


class ChatMessageSerializer(serializers.ModelSerializer):
    """채팅 메시지 시리얼라이저"""

    class Meta:
        model = ChatMessage
        fields = [
            'id',
            'role',
            'content',
            'sources',
            'metadata',
            'created_at'
        ]
        read_only_fields = ['id', 'created_at']


class ChatSessionSerializer(serializers.ModelSerializer):
    """채팅 세션 시리얼라이저"""
    messages = ChatMessageSerializer(many=True, read_only=True)
    message_count = serializers.SerializerMethodField()

    class Meta:
        model = ChatSession
        fields = [
            'id',
            'title',
            'messages',
            'message_count',
            'created_at',
            'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']

    def get_message_count(self, obj):
        return obj.messages.count()


class ChatSessionListSerializer(serializers.ModelSerializer):
    """채팅 세션 목록용 시리얼라이저 (메시지 제외)"""
    message_count = serializers.SerializerMethodField()
    last_message = serializers.SerializerMethodField()

    class Meta:
        model = ChatSession
        fields = [
            'id',
            'title',
            'message_count',
            'last_message',
            'created_at',
            'updated_at'
        ]

    def get_message_count(self, obj):
        return obj.messages.count()

    def get_last_message(self, obj):
        last_msg = obj.messages.last()
        if last_msg:
            return {
                'role': last_msg.role,
                'content': last_msg.content[:100],
                'created_at': last_msg.created_at
            }
        return None


class ChatBookmarkSerializer(serializers.ModelSerializer):
    """북마크 시리얼라이저"""

    class Meta:
        model = ChatBookmark
        fields = [
            'id',
            'content',
            'sources',
            'created_at'
        ]
        read_only_fields = ['id', 'created_at']
```

### 3. Views 개선

**파일 위치:** `backend/apps/chatbot/views.py`

```python
import os
import logging
import requests
from typing import List, Dict, Any

from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import ChatSession, ChatMessage, ChatBookmark
from .serializers import (
    ChatSessionSerializer,
    ChatSessionListSerializer,
    ChatMessageSerializer,
    ChatBookmarkSerializer
)

logger = logging.getLogger(__name__)

# RunPod RAG 서버 URL
RUNPOD_RAG_URL = os.environ.get('RUNPOD_RAG_URL', '')

if not RUNPOD_RAG_URL:
    logger.warning("RUNPOD_RAG_URL environment variable not set!")


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def chat(request):
    """
    사용자 질문 → RunPod RAG → DB 저장

    Request:
        {
            "message": "git rebase란?",
            "session_id": 123,  # optional
            "history": [...]     # optional (legacy support)
        }

    Response:
        {
            "success": true,
            "session_id": 123,
            "message_id": 456,
            "data": {
                "response": "답변 내용",
                "sources": [...]
            }
        }
    """
    message = request.data.get('message')
    session_id = request.data.get('session_id')
    history = request.data.get('history', [])

    if not message:
        return Response({
            'success': False,
            'error': '메시지가 없습니다.'
        }, status=status.HTTP_400_BAD_REQUEST)

    # 1. 채팅 세션 조회 또는 생성
    if session_id:
        try:
            session = ChatSession.objects.get(id=session_id, user=request.user)
        except ChatSession.DoesNotExist:
            return Response({
                'success': False,
                'error': '세션을 찾을 수 없습니다.'
            }, status=status.HTTP_404_NOT_FOUND)
    else:
        # 새 세션 생성
        session = ChatSession.objects.create(
            user=request.user,
            title=message[:100]  # 첫 질문을 제목으로
        )
        logger.info(f"[Chat] New session created: {session.id}")

    # 2. 사용자 메시지 저장
    user_message = ChatMessage.objects.create(
        session=session,
        role='user',
        content=message
    )

    # 3. 채팅 히스토리 구성 (최근 5개 메시지)
    recent_messages = session.messages.order_by('created_at')[:10]
    chat_history = [
        {
            "role": msg.role,
            "content": msg.content
        }
        for msg in recent_messages
        if msg.id != user_message.id  # 현재 메시지 제외
    ]

    # 4. RunPod RAG 호출
    if not RUNPOD_RAG_URL:
        return Response({
            'success': False,
            'error': 'RAG 서버가 설정되지 않았습니다.'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    payload = {
        'question': message,
        'user_id': str(request.user.id),
        'chat_history': chat_history,
        'session_id': str(session.id)
    }

    try:
        logger.info(f"[Chat] Calling RunPod RAG: {RUNPOD_RAG_URL}")

        response = requests.post(
            f"{RUNPOD_RAG_URL}/api/v1/chat",
            json=payload,
            timeout=60  # 60초 타임아웃
        )

        if response.status_code == 200:
            result = response.json()

            if result.get('success'):
                # 5. 응답 메시지 저장
                assistant_message = ChatMessage.objects.create(
                    session=session,
                    role='assistant',
                    content=result.get('answer', ''),
                    sources=result.get('sources', []),
                    metadata=result.get('metadata', {})
                )

                logger.info(f"[Chat] Response saved: message_id={assistant_message.id}")

                return Response({
                    'success': True,
                    'session_id': session.id,
                    'message_id': assistant_message.id,
                    'data': {
                        'response': result.get('answer'),
                        'sources': result.get('sources', [])
                    }
                })
            else:
                error_msg = result.get('error', 'RAG 서버에서 오류가 발생했습니다.')
                logger.error(f"[Chat] RAG server error: {error_msg}")
                return Response({
                    'success': False,
                    'error': error_msg
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            logger.error(f"[Chat] RAG server returned {response.status_code}")
            return Response({
                'success': False,
                'error': f'RAG 서버 오류 (HTTP {response.status_code})'
            }, status=status.HTTP_502_BAD_GATEWAY)

    except requests.exceptions.Timeout:
        logger.error("[Chat] RAG server timeout")
        return Response({
            'success': False,
            'error': 'RAG 서버 응답 시간이 초과되었습니다.'
        }, status=status.HTTP_504_GATEWAY_TIMEOUT)

    except requests.exceptions.ConnectionError:
        logger.error("[Chat] RAG server connection error")
        return Response({
            'success': False,
            'error': 'RAG 서버에 연결할 수 없습니다.'
        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)

    except Exception as e:
        logger.exception(f"[Chat] Unexpected error: {e}")
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_sessions(request):
    """사용자의 모든 채팅 세션 조회"""
    sessions = ChatSession.objects.filter(user=request.user)
    serializer = ChatSessionListSerializer(sessions, many=True)
    return Response({
        'success': True,
        'data': serializer.data
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_session_history(request, session_id):
    """특정 세션의 채팅 내역 조회"""
    try:
        session = ChatSession.objects.get(id=session_id, user=request.user)
        serializer = ChatSessionSerializer(session)
        return Response({
            'success': True,
            'data': serializer.data
        })
    except ChatSession.DoesNotExist:
        return Response({
            'success': False,
            'error': '세션을 찾을 수 없습니다.'
        }, status=status.HTTP_404_NOT_FOUND)


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_session(request, session_id):
    """채팅 세션 삭제"""
    try:
        session = ChatSession.objects.get(id=session_id, user=request.user)
        session.delete()
        return Response({
            'success': True,
            'message': '세션이 삭제되었습니다.'
        })
    except ChatSession.DoesNotExist:
        return Response({
            'success': False,
            'error': '세션을 찾을 수 없습니다.'
        }, status=status.HTTP_404_NOT_FOUND)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_bookmarks(request):
    """사용자의 모든 북마크 조회"""
    bookmarks = ChatBookmark.objects.filter(user=request.user)
    serializer = ChatBookmarkSerializer(bookmarks, many=True)
    return Response({
        'success': True,
        'data': serializer.data
    })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_bookmark(request):
    """북마크 생성"""
    content = request.data.get('content')
    sources = request.data.get('sources', [])

    if not content:
        return Response({
            'success': False,
            'error': '내용이 없습니다.'
        }, status=status.HTTP_400_BAD_REQUEST)

    bookmark = ChatBookmark.objects.create(
        user=request.user,
        content=content,
        sources=sources
    )

    serializer = ChatBookmarkSerializer(bookmark)
    return Response({
        'success': True,
        'data': serializer.data
    }, status=status.HTTP_201_CREATED)


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_bookmark(request, bookmark_id):
    """북마크 삭제"""
    try:
        bookmark = ChatBookmark.objects.get(id=bookmark_id, user=request.user)
        bookmark.delete()
        return Response({
            'success': True,
            'message': '북마크가 삭제되었습니다.'
        })
    except ChatBookmark.DoesNotExist:
        return Response({
            'success': False,
            'error': '북마크를 찾을 수 없습니다.'
        }, status=status.HTTP_404_NOT_FOUND)
```

### 4. URLs 완성

**파일 위치:** `backend/apps/chatbot/urls.py`

```python
from django.urls import path
from . import views

app_name = 'chatbot'

urlpatterns = [
    # 채팅
    path('chat/', views.chat, name='chat'),

    # 세션 관리
    path('sessions/', views.get_sessions, name='sessions'),
    path('sessions/<int:session_id>/', views.get_session_history, name='session-history'),
    path('sessions/<int:session_id>/delete/', views.delete_session, name='delete-session'),

    # 북마크
    path('bookmarks/', views.get_bookmarks, name='bookmarks'),
    path('bookmark/', views.create_bookmark, name='create-bookmark'),
    path('bookmark/<int:bookmark_id>/', views.delete_bookmark, name='delete-bookmark'),
]
```

### 5. 환경변수 설정

**파일 위치:** `backend/.env` 또는 `docker-compose.yml`

```bash
# RunPod RAG 서버 URL
RUNPOD_RAG_URL=https://xxxxx-8080.proxy.runpod.net

# Django Secret Key
SECRET_KEY=your-django-secret-key

# Database
DB_NAME=hint_system
DB_USER=hint_user
DB_PASSWORD=your_password
DB_HOST=db
DB_PORT=3306

# JWT
JWT_SECRET_KEY=your-jwt-secret
```

### 6. 마이그레이션

```bash
# Django 컨테이너에 접속
docker-compose exec backend bash

# 마이그레이션 파일 생성
python manage.py makemigrations chatbot

# 마이그레이션 적용
python manage.py migrate

# 테이블 확인
python manage.py dbshell
SHOW TABLES;
DESC chat_sessions;
DESC chat_messages;
DESC chat_bookmarks;
```

---

## 🧪 배포 및 테스트

### 1. 전체 흐름 테스트

#### Step 1: RunPod RAG 서버 직접 테스트

```bash
# RunPod에서 실행
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "git rebase란 무엇인가요?",
    "user_id": "test_user",
    "chat_history": []
  }'

# 예상 출력:
{
  "success": true,
  "answer": "git rebase는...",
  "sources": [
    {
      "content": "...",
      "url": "https://...",
      "score": null
    }
  ],
  "metadata": {
    "rag_type": "langgraph",
    "response_time": 8.5
  }
}
```

#### Step 2: EC2에서 RunPod 호출 테스트

```bash
# EC2에서 실행 (Django 컨테이너 내부)
curl -X POST https://xxxxx-8080.proxy.runpod.net/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Python async/await 사용법은?",
    "user_id": "user123",
    "chat_history": []
  }'
```

#### Step 3: Django API 테스트

```bash
# 로그인하여 JWT 토큰 얻기
TOKEN=$(curl -X POST http://localhost:8000/api/v1/auth/login/ \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "password"}' \
  | jq -r '.access')

# 채팅 테스트
curl -X POST http://localhost:8000/api/v1/chatbot/chat/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "Docker Compose란?"}'

# 세션 목록 조회
curl -X GET http://localhost:8000/api/v1/chatbot/sessions/ \
  -H "Authorization: Bearer $TOKEN"
```

### 2. 프론트엔드 통합 테스트

React 컴포넌트에서 테스트:

```javascript
// Chatbot/index.jsx
const handleSend = async () => {
  try {
    const response = await api.post('/chatbot/chat/', {
      message: input,
      session_id: currentSessionId  // 기존 세션 ID (없으면 null)
    })

    if (response.data.success) {
      // 새 세션 ID 저장
      if (!currentSessionId) {
        setCurrentSessionId(response.data.session_id)
      }

      // 메시지 추가
      const assistantMessage = {
        role: 'assistant',
        content: response.data.data.response,
        sources: response.data.data.sources
      }
      setMessages(prev => [...prev, assistantMessage])
    }
  } catch (error) {
    console.error('Chat error:', error)
  }
}
```

### 3. 성능 확인

```bash
# RunPod 서버 로그 확인
tail -f /workspace/testrag/experiments/rag_pipeline/server.log

# Django 서버 로그 확인
docker-compose logs -f backend

# MySQL 연결 확인
docker-compose exec db mysql -u hint_user -p hint_system
SELECT * FROM chat_sessions ORDER BY created_at DESC LIMIT 10;
SELECT * FROM chat_messages ORDER BY created_at DESC LIMIT 20;
```

---

## 🐛 문제 해결

### 문제 1: RunPod RAG 서버 연결 실패

**에러:**
```
requests.exceptions.ConnectionError: Failed to establish a new connection
```

**원인:**
- RunPod 포트 포워딩이 안 되어 있음
- RunPod URL이 잘못됨

**해결:**
```bash
# RunPod 대시보드에서 포트 8080 public 설정 확인
# Django .env에서 RUNPOD_RAG_URL 확인
echo $RUNPOD_RAG_URL

# RunPod 서버 실행 확인
ps aux | grep serve_unified
```

### 문제 2: JWT 인증 실패

**에러:**
```
{
  "detail": "Authentication credentials were not provided."
}
```

**해결:**
```javascript
// Axios 인터셉터 확인 (services/api.js)
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('accessToken')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})
```

### 문제 3: CORS 에러

**에러:**
```
Access to XMLHttpRequest has been blocked by CORS policy
```

**해결:**
```python
# Django settings.py
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://your-domain.com",
]
CORS_ALLOW_CREDENTIALS = True
```

### 문제 4: RunPod 서버 타임아웃

**에러:**
```
requests.exceptions.Timeout: Read timed out
```

**해결:**
```python
# views.py에서 타임아웃 증가
response = requests.post(
    f"{RUNPOD_RAG_URL}/api/v1/chat",
    json=payload,
    timeout=120  # 30초 → 120초
)
```

### 문제 5: DB 마이그레이션 실패

**에러:**
```
django.db.utils.OperationalError: (1054, "Unknown column")
```

**해결:**
```bash
# 마이그레이션 초기화
docker-compose exec backend bash
python manage.py migrate chatbot zero
python manage.py makemigrations chatbot
python manage.py migrate chatbot
```

---

## 📊 모니터링

### RunPod 서버 모니터링

```bash
# GPU 사용률
watch -n 1 nvidia-smi

# 서버 로그 실시간 확인
tail -f server.log | grep -E "Chat|Error"

# 프로세스 확인
ps aux | grep python
```

### Django 서버 모니터링

```bash
# 컨테이너 상태
docker-compose ps

# 로그 확인
docker-compose logs -f backend

# DB 쿼리 확인
docker-compose exec db mysql -u hint_user -p
SELECT COUNT(*) FROM chat_messages;
SELECT COUNT(*) FROM chat_sessions;
```

---

## 📝 체크리스트

### RunPod 설정
- [ ] RAG 시스템 실행 확인 (`python serve_unified.py`)
- [ ] Health check 성공 (`curl /api/v1/health`)
- [ ] 포트 8080 public 설정
- [ ] Public URL 복사

### EC2 Django 설정
- [ ] 모델 추가 (`models.py`)
- [ ] Serializers 추가 (`serializers.py`)
- [ ] Views 업데이트 (`views.py`)
- [ ] URLs 완성 (`urls.py`)
- [ ] 환경변수 설정 (`RUNPOD_RAG_URL`)
- [ ] 마이그레이션 완료

### 테스트
- [ ] RunPod 직접 호출 성공
- [ ] EC2 → RunPod 호출 성공
- [ ] Django API 테스트 성공
- [ ] 프론트엔드 통합 테스트 성공
- [ ] DB 저장 확인

---

## 🎯 다음 단계

1. **프로덕션 배포**
   - Nginx HTTPS 설정
   - RunPod 고정 URL 설정
   - 로깅 및 모니터링 강화

2. **성능 최적화**
   - RAG 응답 캐싱
   - DB 쿼리 최적화
   - 비동기 처리 (Celery)

3. **기능 확장**
   - 멀티턴 대화 개선
   - 사용자 피드백 수집
   - A/B 테스트 (Optimized vs LangGraph)

---

**작성:** Claude Code
**날짜:** 2025-12-01
**버전:** 1.0
