# Production Deployment Checklist

## 📋 Pre-Deployment Checklist

### Infrastructure

#### Database
- [ ] **Production Database Setup**
  - [ ] Provision PostgreSQL 16+ instance (AWS RDS, Google Cloud SQL, or DigitalOcean)
  - [ ] Configure automated backups (daily at minimum)
  - [ ] Set up point-in-time recovery (PITR)
  - [ ] Enable connection pooling (PgBouncer or RDS Proxy)
  - [ ] Configure read replicas for dashboard queries
  
- [ ] **Database Tuning**
  ```sql
  -- Tune PostgreSQL settings for production workload
  ALTER SYSTEM SET shared_buffers = '4GB';
  ALTER SYSTEM SET effective_cache_size = '12GB';
  ALTER SYSTEM SET work_mem = '64MB';
  ALTER SYSTEM SET maintenance_work_mem = '1GB';
  ALTER SYSTEM SET max_connections = 200;
  ```

- [ ] **Monitoring**
  - [ ] Set up database monitoring (CloudWatch, Datadog, or Prometheus)
  - [ ] Track: connection pool usage, query latency, deadlocks
  - [ ] Alert on: connection pool exhaustion, slow queries (>1s), high CPU

#### Application Hosting
- [ ] **Choose Deployment Platform**
  - Option A: AWS (ECS/Fargate, Lambda for pipelines)
  - Option B: Google Cloud (Cloud Run, Cloud Functions)
  - Option C: DigitalOcean (App Platform, Kubernetes)
  - Option D: Render/Fly.io (simpler, good for MVP)

- [ ] **Container Setup**
  - [ ] Create production Dockerfile
  - [ ] Optimize image size (multi-stage build)
  - [ ] Security scan container images
  - [ ] Set up container registry (ECR, GCR, Docker Hub)

- [ ] **Environment Configuration**
  - [ ] Production environment variables
  - [ ] Secrets management (AWS Secrets Manager, GCP Secret Manager)
  - [ ] API credentials stored securely
  - [ ] Database credentials rotated regularly

#### Networking
- [ ] **Security**
  - [ ] HTTPS/TLS certificates (Let's Encrypt or cloud provider)
  - [ ] Firewall rules (only allow necessary ports)
  - [ ] VPC/private networking for database
  - [ ] DDoS protection (CloudFlare or cloud provider)

- [ ] **DNS**
  - [ ] Domain name configured
  - [ ] DNS records pointing to production
  - [ ] CDN for static assets (optional)

### Application Configuration

#### Logging
- [ ] **Structured Logging**
  ```python
  # Already using structlog - good!
  # Ensure all logs go to centralized logging
  ```
- [ ] **Log Aggregation**
  - [ ] Set up log aggregation (CloudWatch Logs, Datadog, ELK)
  - [ ] Configure log retention (30 days for debug, 90 days for errors)
  - [ ] Set up log-based alerts

#### Monitoring & Alerting
- [ ] **Application Metrics**
  ```python
  # Already have Prometheus metrics in API - good!
  # Add more:
  # - Pipeline execution times
  # - Signal generation rate
  # - Feature computation latency
  ```

- [ ] **Health Checks**
  ```python
  # Enhance /health endpoint:
  # - Check database connectivity
  # - Verify pipeline status
  # - Check data freshness
  # - Model availability
  ```

- [ ] **Alerting Rules**
  - [ ] Pipeline failures (>2 in 1 hour)
  - [ ] Database connection issues
  - [ ] API error rate >5%
  - [ ] Data staleness (>6 hours old)
  - [ ] High confidence signals (optional)

#### Performance
- [ ] **Database Optimization**
  - [ ] Add missing indexes (check slow query log)
  - [ ] Optimize dashboard queries (add LIMIT clauses - ✅ done!)
  - [ ] Consider materialized views for complex queries
  
- [ ] **Caching**
  - [ ] Redis for frequently accessed data
  - [ ] Cache API responses (markets, signals)
  - [ ] Cache dashboard data (5-15 minute TTL)

- [ ] **Async Processing**
  - [ ] Use Celery or RQ for long-running tasks
  - [ ] Queue system for pipeline orchestration
  - [ ] Background job monitoring

### Security

#### Authentication & Authorization
- [ ] **API Security**
  - [ ] API key authentication for external access
  - [ ] Rate limiting (100 req/min per IP) - use Redis
  - [ ] CORS configuration for dashboard
  
- [ ] **Dashboard Access**
  - [ ] SSO or OAuth (Google, GitHub)
  - [ ] Role-based access control (admin vs read-only)
  - [ ] Session management

#### Data Protection
- [ ] **Encryption**
  - [ ] Encryption at rest (database)
  - [ ] Encryption in transit (TLS)
  - [ ] Secrets encrypted in environment

- [ ] **SQL Injection Prevention** ✅
  - Already fixed in code review
  - Verify all queries use parameterized queries

- [ ] **Input Validation**
  - [ ] Validate all user inputs
  - [ ] Sanitize file uploads (if any)
  - [ ] Rate limit expensive operations

### CI/CD

#### Automated Testing
- [ ] **GitHub Actions Workflow**
  ```yaml
  # .github/workflows/test.yml
  name: Tests
  on: [push, pull_request]
  jobs:
    test:
      runs-on: ubuntu-latest
      services:
        postgres:
          image: postgres:16
          env:
            POSTGRES_PASSWORD: test
          options: >-
            --health-cmd pg_isready
            --health-interval 10s
      steps:
        - uses: actions/checkout@v3
        - name: Run tests
          run: |
            pip install -r requirements-core.txt -r requirements-dev.txt
            pytest -v
        - name: Lint
          run: ruff check .
  ```

- [ ] **Test Coverage**
  - [ ] Maintain >80% coverage
  - [ ] Integration tests pass
  - [ ] Live API tests (manual or scheduled)

#### Deployment Pipeline
- [ ] **Staging Environment**
  - [ ] Deploy to staging before production
  - [ ] Run smoke tests in staging
  - [ ] Manual approval gate for production

- [ ] **Deployment Strategy**
  - [ ] Blue-green deployment (zero downtime)
  - [ ] Automatic rollback on failure
  - [ ] Database migrations run automatically

- [ ] **Post-Deployment**
  - [ ] Health check verification
  - [ ] Smoke tests
  - [ ] Monitor error rates for 1 hour

### Data Management

#### Database Migrations
- [ ] **Alembic Setup** ✅
  - Already configured
  - Ensure migrations are tested

- [ ] **Backup Strategy**
  - [ ] Daily automated backups
  - [ ] Weekly full backups
  - [ ] Test restore quarterly
  - [ ] RTO: 1 hour, RPO: 1 hour

#### Data Retention
- [ ] **Cleanup Jobs**
  ```python
  # Delete old ticks (keep 90 days)
  DELETE FROM odds_ticks WHERE tick_ts < NOW() - INTERVAL '90 days';
  
  # Archive old backtest runs
  DELETE FROM backtest_runs WHERE created_at < NOW() - INTERVAL '180 days';
  ```

- [ ] **Archive Strategy**
  - [ ] Archive old data to S3/GCS
  - [ ] Compress archived data
  - [ ] Document retention policy

### Documentation

#### Operational Docs
- [ ] **Runbooks**
  - [ ] How to restart services
  - [ ] How to restore from backup
  - [ ] How to scale resources
  - [ ] Common troubleshooting steps

- [ ] **Architecture Diagram**
  - [ ] System architecture
  - [ ] Data flow
  - [ ] External dependencies

#### User Documentation
- [ ] **API Documentation** ✅
  - Already have OpenAPI docs
  - Keep up to date

- [ ] **Dashboard Guide**
  - [ ] Feature walkthrough
  - [ ] Common workflows
  - [ ] Interpretation guide

### Cost Optimization

#### Infrastructure Costs
- [ ] **Database**
  - [ ] Right-size instance (start small, scale up)
  - [ ] Use reserved instances for savings
  - [ ] Monitor unused indexes

- [ ] **Compute**
  - [ ] Use spot instances for pipelines (if possible)
  - [ ] Auto-scaling based on load
  - [ ] Shutdown non-production environments at night

- [ ] **Storage**
  - [ ] Lifecycle policies for old data
  - [ ] Use cheaper storage tiers for archives
  - [ ] Monitor storage growth

#### API Costs
- [ ] **Prediction Markets**
  - [ ] Document Kalshi/Polymarket API costs
  - [ ] Set rate limits to control costs
  - [ ] Cache responses where possible

### Compliance

#### Financial Regulations
- [ ] **Disclaimers**
  - [ ] "Not financial advice"
  - [ ] "Past performance doesn't guarantee future results"
  - [ ] Risk disclosures

- [ ] **Audit Trail**
  - [ ] Log all signal generation
  - [ ] Log all trades (even paper trades)
  - [ ] Retain logs per regulatory requirements

#### Privacy
- [ ] **Privacy Policy**
  - [ ] Document data collection
  - [ ] User consent for tracking
  - [ ] Data retention policy

- [ ] **GDPR Compliance** (if applicable)
  - [ ] Right to access data
  - [ ] Right to deletion
  - [ ] Data portability

## 🚀 Launch Checklist

### Week Before Launch
- [ ] Load testing completed
- [ ] Security audit passed
- [ ] Backup/restore tested
- [ ] Monitoring dashboards set up
- [ ] On-call schedule created

### Day Before Launch
- [ ] Final code freeze
- [ ] Staging environment tested
- [ ] Database migrations tested
- [ ] Rollback plan documented
- [ ] Team briefed on launch plan

### Launch Day
- [ ] Deploy to production (off-peak hours)
- [ ] Verify health checks pass
- [ ] Run smoke tests
- [ ] Monitor for 2 hours
- [ ] Announce launch (if public)

### Week After Launch
- [ ] Monitor error rates daily
- [ ] Gather user feedback
- [ ] Fix critical bugs immediately
- [ ] Document lessons learned

## 📊 Success Metrics

### Technical
- [ ] 99.9% uptime
- [ ] <1s API response time (p95)
- [ ] <5% error rate
- [ ] Zero data loss

### Business
- [ ] Pipeline runs successfully daily
- [ ] Signals generated correctly
- [ ] Users find value in dashboard
- [ ] Costs within budget

## 🆘 Emergency Contacts

- **Database Issues**: [DBA contact]
- **Infrastructure Issues**: [DevOps contact]
- **API Issues**: [Backend lead]
- **Security Issues**: [Security team]

## 📚 Resources

- Production URL: [TBD]
- Monitoring Dashboard: [TBD]
- Log Aggregation: [TBD]
- Status Page: [TBD]
- Documentation: [TBD]

