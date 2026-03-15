# MBH_App
Web application to visualize and classify hemorrhages in CT images.

## Prerequisites
Before running or deploying this project, make sure you have the following installed:
- [Node.js & npm](https://nodejs.org/)
- [Docker](https://www.docker.com/) & [LocalStack](https://localstack.cloud/) (for local AWS simulation)
- [Terraform](https://www.terraform.io/)
- [awscli-local](https://github.com/localstack/awscli-local) (`awslocal`)

---

## Local Development
To work on the app locally with hot-reloading:

1. Install dependencies:
   ```bash
   npm install
   ```
2. Start the development server:
   ```bash
   npm run dev
   ```
3. Open `http://localhost:5173` in your browser.

---

## Build for Production
To bundle the application into static files for deployment:
```bash
npm run build
```
This will create a `dist/` folder containing your optimized HTML, CSS, and JS files.

---

## Local Cloud Deployment (AWS S3 via LocalStack)
This project includes an Infrastructure as Code (IaC) setup using Terraform to host the site securely on an S3 bucket.

1. **Start LocalStack:** Ensure your LocalStack Docker container is running on port 4566.
2. **Build the Infrastructure:**
   ```bash
   cd terraform
   terraform init
   terraform apply
   ```
   *(Type `yes` when prompted to create the S3 bucket and IAM policies).*
3. **Upload the Website:**
   Navigate back to the root folder and sync the `dist/` folder to the S3 bucket:
   ```bash
   cd ..
   awslocal s3 sync ./dist s3://my-vite-app-bucket
   ```
4. **View the Live App:**
   Open the following URL in your browser:
   `http://my-vite-app-bucket.s3-website.localhost.localstack.cloud:4566`


